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

#define min(x, y) (x)<(y)?(x):(y)
#define max(x, y) (x)>(y)?(x):(y)

template<class T>
inline T clip(float x, T _min, T _max){
    return (T)fmax(fmin(x, _max), _min);
}

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

int main(int argc, char** argv) {

    if(argc != 6){
        std::cout << "xxx image prompt num_inference_steps strength controlnet_conditioning_scale\n";
        return 0;
    }

    auto image = cv::imread(argv[1]);
    std::string trigger = argv[2];
    auto num_inference_steps = atoi(argv[3]);//10;

    auto strength = atof(argv[4]); // 0~1 之间重绘比例。越低越接近输入图片。
    float controlnet_conditioning_scale = atof(argv[5]); // 0~1 之间的 ControlNet 约束比例。越高越贴近约束。

    std::vector<float> noise(1 * 4 * 64 * 64);
    AIDB::Utility::randn(noise.data(), noise.size());

    //    strength: This parameter will take a value between 0 and 1.
    //    The higher the value less the final image is going to look similar to the seed image.
    // do_classifier_free_guidance
    // For classifier free guidance, we need to do two forward passes.
    auto scheduler = Scheduler::DDIMScheduler("./config/scheduler_config.json");
    auto scaling_factor = 0.18215f;
    auto tokenizer = Tokenizer::FromBlobJSON(
            LoadBytesFromFile("./config/tokenizer.json"));
    std::string startoftext = "<|startoftext|>";
    std::string endoftext = "<|endoftext|>";
//    std::string prompt = startoftext + "shinkawa youji" + endoftext;

    std::string prompt = startoftext + trigger + endoftext;
    std::vector<int> text_input_ids = tokenizer->Encode(prompt);

    std::string uncond_tokens = startoftext + "longbody, lowres, cropped, worst quality, low quality, multiple people" + endoftext;

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

    // Figuring initial time step based on strength
    auto init_timestep = min(int(num_inference_steps * strength), num_inference_steps);
    auto t_start = max(num_inference_steps - init_timestep, 0);

    timesteps.assign(timesteps.begin() + t_start, timesteps.end());

    num_inference_steps = timesteps.size();

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

    auto low_threshold = 150;
    auto high_threshold = 200;
    cv::Mat canny;
    cv::Canny(image, canny, low_threshold, high_threshold);

    std::vector<cv::Mat> bgr_channels{canny, canny, canny};
    cv::merge(bgr_channels, canny);

    image.convertTo(image, CV_32F);
    image = image / 127.5 - 1.0;
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);

    canny.convertTo(canny, CV_32F);
//    canny = canny / 127.5 - 1.0;
    cv::Mat blob_canny;
    cv::dnn::blobFromImage(canny, blob_canny, 1.0f / 255.0f);

    auto vae_enc = AIDB::Interpreter::createInstance("sd_vae_encoder_with_controlnet", "onnx");

    if(nullptr == vae_enc){
        return -1;
    }

    auto vae_dec = AIDB::Interpreter::createInstance("sd_vae_decoder_with_controlnet", "onnx");

    if(nullptr == vae_dec){
        return -1;
    }

    auto unet = AIDB::Interpreter::createInstance2("sd_unet_with_controlnet_with_lora", "shinkawa", "onnx");

    if(nullptr == unet){
        return -1;
    }

    // Prepare latent variables
    std::vector<std::vector<float>>  image_latents;
    std::vector<std::vector<int>>  image_latents_shape;

    vae_enc->forward(blob.data, 512, 512, 3, image_latents, image_latents_shape);

    auto latents = image_latents[0];


    std::for_each(latents.begin(), latents.end(), [=](float &item){item *= scaling_factor;});

    auto latent_timestep = timesteps[0];
    std::vector<float> init_latents;
    scheduler.add_noise(latents, {1, 4, 64, 64}, noise, {1, 4, 64, 64}, latent_timestep, init_latents);

    auto guidance_scale = 7.5f;

    std::vector<float> controlnet_keep(timesteps.size(), 1.0);

    int step = 0;
    for(auto t: timesteps){
        auto tic = std::chrono::system_clock::now();
        double cond_scale = controlnet_conditioning_scale * controlnet_keep[step];
        std::vector<float> latent_model_input(2 * 4 * 64 * 64, 0);
        memcpy(latent_model_input.data(), init_latents.data(), 4 * 64 * 64 * sizeof(float));
        memcpy(latent_model_input.data() + 4 * 64 * 64, init_latents.data(), 4 * 64 * 64 * sizeof(float));


        std::vector<std::vector<float>> down_and_mid_blok_samples;
        std::vector<std::vector<int>> down_and_mid_blok_samples_shape;
        std::vector<void*> input;
        std::vector<std::vector<int>> input_shape;

        // sample
        input.push_back(latent_model_input.data());
        input_shape.push_back({2, 4, 64, 64});


        // t ✅
        std::vector<float> timestep = {(float)t};
        input.push_back(timestep.data());
        input_shape.push_back({1});

        // encoder_hidden_states ✅
        input.push_back(prompt_embeds_cat.data());
        input_shape.push_back({2, 77, 768});

        std::vector<float> controlnet_cond(2 * 3 * 512 * 512, 0);
        memcpy(controlnet_cond.data(), blob_canny.data, 3 * 512 * 512 * sizeof(float));
        memcpy(controlnet_cond.data() + 3 * 512 * 512, blob_canny.data, 3 * 512 * 512 * sizeof(float));


        // controlnet_cond ✅
        input.push_back(controlnet_cond.data());
        input_shape.push_back({2, 3, 512, 512});

        // conditioning_scale ✅
        std::vector<float> cond_scales = {(float)(cond_scale)};
        input.push_back(cond_scales.data());
        input_shape.push_back({1});

        std::vector<std::vector<float>> noise_preds;
        std::vector<std::vector<int>> noise_preds_shape;
        unet->forward(input, input_shape, noise_preds, noise_preds_shape);

        // noise_preds [2,4,64,64] noise_pred_uncond | noise_pred_text
        std::vector<float> noise_pred(1 * 4 * 64 * 64, 0);
        for(int i = 0; i < noise_pred.size(); i++){
            noise_pred[i] = noise_preds[0][i] + guidance_scale * (noise_preds[0][i + 4 * 64 * 64] - noise_preds[0][i]);
        }
        std::vector<float> pred_sample;
        scheduler.step(noise_pred, {1, 4, 64, 64}, init_latents, {1, 4, 64, 64}, pred_sample, t);
        init_latents.clear();
        init_latents.assign(pred_sample.begin(), pred_sample.end());
        auto toc = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = toc - tic;
        // 计算并输出进度百分比
        std::cout << "\rStep " << step++ << " " <<std::fixed << std::setprecision(1) << float(step) / timesteps.size() * 100
                  << "% " << std::fixed << std::setprecision(3) << 1.0f / elapsed.count() << "b/s"
                  << " [" << std::setw(float(step) / (float)timesteps.size() * 30) << std::setfill('=') << '>' << ']';
        std::flush(std::cout); // 刷新输出缓冲区，确保立即显示进度
    }
    std::for_each(init_latents.begin(), init_latents.end(), [=](float &item){item /= scaling_factor;});

    std::vector<std::vector<float>>  sample;
    std::vector<std::vector<int>>  sample_shape;
    vae_dec->forward(init_latents.data(), 64, 64, 4, sample, sample_shape);

    cv::Mat sd_image(sample_shape[0][2], sample_shape[0][3], CV_8UC3);
    AIDB::Utility::stable_diffusion_process(sample[0].data(), sample_shape[0][2], sample_shape[0][3], sample_shape[0][1], sd_image);
    cv::imwrite("stable_diffusion_controlnet_img2img_" + trigger + ".jpg", sd_image);
    return 0;
}
