//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"

void test_yolov8(AIDB::Interpreter* ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *ins << bgr;

    cv::Mat src_image;
    *ins >> src_image;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
    std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

    assert(input->scale_h() == input->scale_w());
    assert(1 == outputs_shape[0][0]);

    AIDB::Utility::yolov8_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.35, ins->scale_h());

    AIDB::Utility::Common::draw_objects(src_image, result, object);
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
        test_yolov8(interpreter, bgr, result);
        cv::imshow("YoloV8", result);
        cv::waitKey();
    } else {
        cv::VideoCapture cap;
//        cv::VideoWriter writer;
//        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        if(input_file.size() < 2)
            cap.open(std::atoi(input_file.c_str()));
        else
            cap.open(input_file);

//        writer.open("yolov8.mp4", codec, 25, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            test_yolov8(interpreter, bgr, result);
//            writer << result;
            cv::imshow("YoloV8", result);
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