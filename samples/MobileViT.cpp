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


int main(int argc, char** argv){
    if(argc != 4){
        std::cout << "xxx model backend image_file\n";
        return 0;
    }
    auto interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == interpreter){
        return -1;
    }


    std::string interpreter_file = argv[3];

    auto post_process = AIDB::Utility::ImageNet("extra/imagenet-1k-id.txt");

    cv::Mat blob = *interpreter << interpreter_file;

    cv::Mat src_image;
    *interpreter >> src_image;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    interpreter->forward((float*)blob.data, interpreter->width(), interpreter->height(), interpreter->channel(), outputs, outputs_shape);

    std::vector<std::shared_ptr<AIDB::ClsMeta>> predicts;

    int topK = 3;
    post_process(outputs[0], outputs_shape[0], predicts, topK);

    std::cout << "topK=" <<topK << " results:\n\t";
    for(int i = 0; i < topK; i++){
        std::cout << i << ": " << "label: [" << predicts[i]->label_str << "] conf: [" << predicts[i]->conf << "]\n\t";
    }

#if __linux__
    // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("src_image", src_image);
            cv::waitKey();
        }
#else
    cv::imshow("src_image", src_image);
    cv::waitKey();
#endif

    return 0;
}