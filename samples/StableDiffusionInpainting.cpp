//
// Created by TalkUHulk on 2024/6/7.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include <chrono>
#include <tokenizers_cpp.h>
#include <fstream>
#include "ddimscheduler.hpp"
#if __linux__
#include <fstream>
#endif

using tokenizers::Tokenizer;


std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}


int main(int argc, char** argv){

    if(argc != 5){
        std::cout << "xxx image mask prompt num_inference_steps\n";
        return 0;
    }

    auto image = cv::imread(argv[1]);
    auto mask = cv::imread(argv[2], 0);
//    auto image = cv::imread("dog.png");
//    auto mask = cv::imread("dog_mask.png", 0);

    std::string trigger = argv[3];

    auto num_inference_steps = atoi(argv[4]);//10;

    auto scheduler = Scheduler::DDIMScheduler("./config/scheduler_config.json");

    auto tokenizer = Tokenizer::FromBlobJSON(
            LoadBytesFromFile("./config/tokenizer.json"));
    std::string startoftext = "<|startoftext|>";
    std::string endoftext = "<|endoftext|>";

    std::string prompt = startoftext + trigger + endoftext;
    std::vector<int> text_input_ids = tokenizer->Encode(prompt);

    std::string uncond_tokens = startoftext + "" + endoftext;

    std::vector<int> uncond_input = tokenizer->Encode(uncond_tokens);

    auto text_enc = AIDB::Interpreter::createInstance("text_encoder", "onnx");

    if(nullptr == text_enc){
        return -1;
    }

    std::vector<std::vector<float>>  prompt_embeds;
    std::vector<std::vector<int>>  prompt_embeds_shape;

    text_enc->forward(text_input_ids.data(), 77, 0, 0,  prompt_embeds, prompt_embeds_shape);

    std::vector<std::vector<float>>  negative_prompt_embeds;
    std::vector<std::vector<int>>  negative_prompt_embeds_shape;
    text_enc->forward(uncond_input.data(), 77, 0, 0,  negative_prompt_embeds, negative_prompt_embeds_shape);

    std::vector<float> prompt_embeds_cat(2 * 77 * 768, 0);
    memcpy(prompt_embeds_cat.data(), negative_prompt_embeds[0].data(), 77 * 768 * sizeof(float));
    memcpy(prompt_embeds_cat.data() + 77 * 768, prompt_embeds[0].data(), 77 * 768 * sizeof(float));

    scheduler.set_timesteps(num_inference_steps);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);

    auto vae_enc = AIDB::Interpreter::createInstance("sd_inpaint_vae_encoder", "onnx");

    if(nullptr == vae_enc){
        return -1;
    }

    auto vae_dec = AIDB::Interpreter::createInstance("sd_inpaint_vae_decoder", "onnx");

    if(nullptr == vae_dec){
        return -1;
    }

    auto unet = AIDB::Interpreter::createInstance("sd_inpaint_unet", "onnx");

    if(nullptr == unet){
        return -1;
    }

    std::vector<float> latents(1 * 4 * 64 * 64);

    AIDB::Utility::randn(latents.data(), latents.size());

    int target = 512;
    float src_ratio = float(image.cols) / float(image.rows);
    float target_ratio = 1.0f;

    int n_w, n_h, pad_w = 0, pad_h = 0;
    float _scale_h, _scale_w;

    if(src_ratio > target_ratio){
        n_w = target;

        n_h = floor(float(n_w) / float(image.cols) * float(image.rows) + 0.5f);
        pad_h = target - n_h;
        _scale_h = _scale_w = float(n_w) / float(image.cols);
    } else if(src_ratio < target_ratio){
        n_h = target;
        n_w = floor(float(n_h) / float(image.rows) * float(image.cols) + 0.5f);
        pad_w = target - n_w;
        _scale_h = _scale_w = float(n_h) / float(image.rows);
    } else{
        n_w = target;
        n_h = target;
        _scale_h = _scale_w = float(n_w) / float(image.cols);
    }

    cv::resize(image, image, cv::Size(n_w, n_h));
    cv::copyMakeBorder(image, image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    cv::resize(mask, mask, cv::Size(n_w, n_h));
    cv::copyMakeBorder(mask, mask, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    cv::threshold(mask, mask, 127.5, 1, cv::THRESH_BINARY);

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F);
    image = image / 127.5 - 1.0;
    cv::Mat mask_image = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
    image.copyTo(mask_image, 1 - mask);

    cv::Mat blob;
    cv::Mat blob_mask;
    cv::dnn::blobFromImage(mask_image, blob);

    cv::dnn::blobFromImage(mask, blob_mask, 1.0f, cv::Size(64, 64));

    std::vector<std::vector<float>>  masked_image_latents;
    std::vector<std::vector<int>>  masked_image_latents_shape;

    vae_enc->forward(blob.data, 512, 512, 3, masked_image_latents, masked_image_latents_shape);

    auto scaling_factor = 0.18215f;
    std::for_each(masked_image_latents.begin(), masked_image_latents.end(),
                  [=](std::vector<float>& masked_image_latent) {
                      std::for_each(masked_image_latent.begin(), masked_image_latent.end(), [=](float &item){ item *= scaling_factor;});
                  }
    );

    auto init_noise_sigma = scheduler.get_init_noise_sigma();
    std::for_each(latents.begin(), latents.end(), [=](float &item){item*=init_noise_sigma;});
    auto guidance_scale = 7.5f;
    int step = 0;
    for(auto t: timesteps){
        auto tic = std::chrono::system_clock::now();

        std::vector<float> latent_model_input(2 * 9 * 64 * 64, 0);
        memcpy(latent_model_input.data(), latents.data(), 4 * 64 * 64 * sizeof(float));
        memcpy(latent_model_input.data() + 4 * 64 * 64, blob_mask.data, 1 * 64 * 64 * sizeof(float));
        memcpy(latent_model_input.data() + 5 * 64 * 64, masked_image_latents[0].data(), 4 * 64 * 64 * sizeof(float));
        memcpy(latent_model_input.data() + 9 * 64 * 64, latent_model_input.data(), 9 * 64 * 64 * sizeof(float));

        std::vector<std::vector<float>> noise_preds;
        std::vector<std::vector<int>>  noise_preds_shape;
        std::vector<void *> input;
        std::vector<std::vector<int>> input_shape;

        input.push_back(latent_model_input.data());
        input_shape.push_back({2, 9, 64, 64});

        std::vector<long long> timestep = {(long long)t};
        input.push_back(timestep.data());
        input_shape.push_back({1});

        input.push_back(prompt_embeds_cat.data());
        input_shape.push_back({2, 77, 768});

        unet->forward(input, input_shape, noise_preds, noise_preds_shape);

        // noise_preds [2,4,64,64] noise_pred_uncond | noise_pred_text
        std::vector<float> noise_pred(1 * 4 * 64 * 64, 0);
        for(int i = 0; i < noise_pred.size(); i++){
            noise_pred[i] = noise_preds[0][i] + guidance_scale * (noise_preds[0][i + 4 * 64 * 64] - noise_preds[0][i]);
        }
        std::vector<float> pred_sample;
        scheduler.step(noise_pred, {1, 4, 64, 64}, latents, {1, 4, 64, 64}, pred_sample, t);
        latents.clear();
        latents.assign(pred_sample.begin(), pred_sample.end());

        auto toc = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = toc - tic;
        // 计算并输出进度百分比
        std::cout << "\rStep " << step++ << " " <<std::fixed << std::setprecision(1) << float(step) / timesteps.size() * 100
                  << "% " << std::fixed << std::setprecision(3) << 1.0f / elapsed.count() << "b/s"
                  << " [" << std::setw(float(step) / (float)timesteps.size() * 30) << std::setfill('=') << '>' << ']';
        std::flush(std::cout); // 刷新输出缓冲区，确保立即显示进度
    }

    std::for_each(latents.begin(), latents.end(), [=](float &item){item /= scaling_factor;});

    std::vector<std::vector<float>>  sample;
    std::vector<std::vector<int>>  sample_shape;
    vae_dec->forward(latents.data(), 64, 64, 4, sample, sample_shape);
    cv::Mat sd_image;
    AIDB::Utility::stable_diffusion_process(sample[0].data(), sample_shape[0][2], sample_shape[0][3], sample_shape[0][1], sd_image);

    cv::imwrite("stable_diffusion_inpainting.jpg", sd_image);
    return 0;
}