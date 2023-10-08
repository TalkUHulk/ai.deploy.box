//
// Created by TalkUHulk on 2023/10/3.
//


#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#if __linux__
#include <fstream>
#endif

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage, double cost) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r\b%3d%% %3.3fs [%.*s%*s]", val, cost, lpad, PBSTR, rpad, "");
    fflush(stdout);
}


int avd_network(AIDB::Interpreter* ins, void* data1, void* data2, std::vector<float> &kp_norm){

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    std::vector<void*> inputs{data1, data2};

    std::vector<std::vector<int>> input_shapes(2, {1, 50, 2});
    ins->forward(inputs, input_shapes, outputs, outputs_shape);

    kp_norm.assign(outputs[0].begin(), outputs[0].end());

    return 0;
}

int kp_detect(AIDB::Interpreter* ins, const cv::Mat& bgr, cv::Mat& blob, std::vector<float> &kp_source){
    blob = *ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

    kp_source.assign(outputs[0].begin(), outputs[0].end());

    return 0;
}

int main(int argc, char** argv){

//    if(argc != 5){
//        std::cout << "xxx model backend image/camera image_file/camera id/video file\n";
//        return 0;
//    }
    auto interpreter_kp = AIDB::Interpreter::createInstance("tpsmm_kp_detector", "openvino");
    auto interpreter_avd = AIDB::Interpreter::createInstance("tpsmm_avd_network", "openvino");
    auto interpreter_tpsmm = AIDB::Interpreter::createInstance("tpsmm", "openvino");

    if(nullptr == interpreter_kp || nullptr == interpreter_avd || nullptr == interpreter_tpsmm){
        return -1;
    }

    auto source = cv::imread("/Users/hulk/Documents/CodeZoo/Thin-Plate-Spline-Motion-Model-main/assets/source.png");
    auto cap = cv::VideoCapture("/Users/hulk/Documents/CodeZoo/Thin-Plate-Spline-Motion-Model-main/assets/driving.mp4");
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    auto video_writer = cv::VideoWriter("aidb_tpsmm_openvino.mp4", codec, 25, cv::Size(256, 256));
    int frame_cnt = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    // source kp
    cv::Mat blob_source;
    std::vector<float> kp_source;
    kp_detect(interpreter_kp, source, blob_source, kp_source);


    int mode = 2;
    bool is_first = true;
    cv::Mat driving;
    std::vector<float> kp_driving_initial;

    std::vector<std::vector<int>> input_shapes = {{1, 50, 2}, {1, 50, 2}, {1, 3, 256, 256}, {1, 3, 256, 256}};

    int cnt = 0;
    while(true) {
        cap >> driving;
        if (driving.empty()) {
            break;
        }
        cnt ++;
        auto tic = std::chrono::high_resolution_clock::now();

        std::vector<float> kp_norm;
        std::vector<float> kp_driving;
        cv::Mat blob_driving;

        if(is_first){
            kp_detect(interpreter_kp, driving, blob_driving, kp_driving_initial);
            kp_driving = kp_driving_initial;
            is_first = false;
        } else{
            kp_detect(interpreter_kp, driving, blob_driving, kp_driving);
        }

        // kp norm
        switch (mode) {
            case 0:{
                kp_norm = kp_driving;
                break;
            }
            case 1:{
                AIDB::Utility::relative_kp(kp_source, kp_driving, kp_driving_initial, kp_norm);
                break;
            }
            case 2:{
                avd_network(interpreter_avd, kp_source.data(), kp_driving.data(), kp_norm);
                break;
            }
            default:{
                std::cout << "not support\n";
            }

        }

        //tpsmm
        std::vector<void*> inputs{kp_source.data(), kp_driving.data(), blob_source.data, blob_driving.data};
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        interpreter_tpsmm->forward(inputs, input_shapes, outputs, outputs_shape);

        int siz[] = {1, 3, 256, 256};

        cv::Mat blob_output(4, siz, CV_32F, outputs[0].data());
        std::vector<cv::Mat> generated;

        cv::dnn::imagesFromBlob(blob_output, generated);

        auto toc = std::chrono::high_resolution_clock::now();    //结束时间
        std::chrono::duration<double> elapsed = toc - tic;

        printProgress(cnt / double(frame_cnt), elapsed.count());
        cv::Mat generatedRGB;
        generated[0].convertTo(generatedRGB, CV_8U,  255);
        cv::cvtColor(generatedRGB, generatedRGB, cv::COLOR_BGR2RGB);
        video_writer << generatedRGB;
//        cv::imshow("demo", generatedRGB);
//        cv::waitKey();
    }

    cap.release();

    video_writer.release();

    AIDB::Interpreter::releaseInstance(interpreter_kp);
    AIDB::Interpreter::releaseInstance(interpreter_avd);
    AIDB::Interpreter::releaseInstance(interpreter_tpsmm);

    return 0;
}