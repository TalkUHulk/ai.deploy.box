//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#if __linux__
#include <fstream>
#endif


void animegan(AIDB::Interpreter* ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *ins << bgr;

    cv::Mat src_image;
    *ins >> src_image;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

    AIDB::Utility::animated_gan_post_process(outputs[0], outputs_shape[0], result);
}

int main(int argc, char** argv){

    if(argc != 5){
        std::cout << "xxx model backend image/camera image_file/camera id/video file\n";
        return 0;
    }
    auto interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == interpreter){
        return -1;
    }

    int input_type = std::atoi(argv[3]);
    std::string input_file = argv[4];


    if(0 == input_type){
        auto bgr = cv::imread(input_file);
        cv::Mat result;
        animegan(interpreter, bgr, result);

        cv::imwrite("AnimeGan.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("AnimeGan", result);
            cv::waitKey();
        }
#else
        cv::imshow("AnimeGan", result);
        cv::waitKey();
#endif
    } else {
        cv::VideoCapture cap;
        if(input_file.size() < 2)
            cap.open(std::atoi(input_file.c_str()));
        else
            cap.open(input_file);

        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            animegan(interpreter, bgr, result);
            cv::imshow("AnimeGan", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();

    }


    AIDB::Interpreter::releaseInstance(interpreter);

    return 0;
}