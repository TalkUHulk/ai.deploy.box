//
// Created by TalkUHulk on 2023/9/4.
//

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

    if(argc != 7){
        std::cout << "xxx model backend image/camera image_file/camera id/video file\n";
        return 0;
    }
    auto encoder_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);
    auto mb_sam_interpreter = AIDB::Interpreter::createInstance(argv[3], argv[4]);

    if(nullptr == encoder_interpreter || nullptr == mb_sam_interpreter){
        return -1;
    }

    int input_type = std::atoi(argv[5]);
    std::string input_file = argv[6];


    auto bgr = cv::imread(input_file);

    cv::Mat blob = *encoder_interpreter << bgr;

    cv::Mat src_image;
    *encoder_interpreter >> src_image;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    encoder_interpreter->forward((float*)blob.data, encoder_interpreter->width(), encoder_interpreter->height(), encoder_interpreter->channel(), outputs, outputs_shape);

    std::vector<void *> input;
    // MNN freeze 3 points, dynamic inputs reshape error.
    std::vector<float> coords{400, 400, 100, 500, 250, 250}; // 1xNx2
    std::vector<float> labels{1, 1, 1}; // 1xN
    std::vector<float> mask_input(256 * 256, 0);
    std::vector<int64_t> input_dim_has{1};

    std::for_each(coords.begin(), coords.end(), [=](float& p){ p *= encoder_interpreter->scale_w();});

    input.emplace_back(outputs[0].data());
    input.emplace_back(coords.data());
    input.emplace_back(labels.data());
    input.emplace_back(mask_input.data());
    input.emplace_back(input_dim_has.data());

    std::vector<std::vector<int>> input_shape;
    input_shape.push_back({1, 256, 64, 64});
    input_shape.push_back({1, 3, 2});
    input_shape.push_back({1, 3});
    input_shape.push_back({1, 1, 256, 256});
    input_shape.push_back({1});

    mb_sam_interpreter->forward(input, input_shape, outputs, outputs_shape);

    cv::Mat result;
    AIDB::Utility::mobile_sam_post_process(outputs[0], src_image, result, encoder_interpreter->scale_w(), cv::Scalar(255, 144, 30));

    for(int i = 0;i < coords.size() / 2; i++){
        cv::circle(result,
                   cv::Point(coords[2 * i] / encoder_interpreter->scale_w(), coords[2 * i + 1] / encoder_interpreter->scale_w()),
                   3, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("MobileSAM", result);
    cv::waitKey();

    AIDB::Interpreter::releaseInstance(encoder_interpreter);
    AIDB::Interpreter::releaseInstance(mb_sam_interpreter);

    return 0;
}