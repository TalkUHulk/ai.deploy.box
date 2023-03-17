//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"

void face_parsing(AIDB::Interpreter* ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *ins << bgr;

    cv::Mat src_image;
    *ins >> src_image;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

    AIDB::Utility::bisenet_post_process(src_image, result, outputs[0], outputs_shape[0]);
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
        face_parsing(interpreter, bgr, result);
        cv::imshow("FaceParsing", result);
        cv::waitKey();
    } else {
        cv::VideoCapture cap;
//        cv::VideoWriter writer;
//        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if(input_file.size() < 2)
            cap.open(std::atoi(input_file.c_str()));
        else
            cap.open(input_file);

//        writer.open("face_parsing.mp4", codec, 25, cv::Size(512, 512));
        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            face_parsing(interpreter, bgr, result);
//            writer << result;

            cv::imshow("FaceParsing", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
//        writer.release();
    }


    AIDB::Interpreter::releaseInstance(interpreter);

    return 0;
}