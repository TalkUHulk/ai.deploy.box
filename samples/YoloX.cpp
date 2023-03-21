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


void test_yolox(AIDB::Interpreter* ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *ins << bgr;

    cv::Mat src_image;
    *ins >> src_image;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
    std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

    assert(ins->scale_h() == ins->scale_w());

    auto post_process = AIDB::Utility::YoloX(ins->width(), 0.25, 0.45, {8, 16, 32});

    post_process(outputs[0], outputs_shape[0], object, src_image.cols, src_image.rows, ins->scale_h());

    AIDB::Utility::Common::draw_objects(src_image, result, object);
}

int main(int argc, char** argv){

    // xxx model backend image/camera image_file/camera id/video file
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
        test_yolox(interpreter, bgr, result);
        cv::imwrite("YoloX.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("YoloX", result);
            cv::waitKey();
        }
#else
        cv::imshow("YoloX", result);
        cv::waitKey();
#endif
    } else {
        cv::VideoCapture cap;
//        cv::VideoWriter writer;
//        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        if(input_file.size() < 2)
            cap.open(std::atoi(input_file.c_str()));
        else
            cap.open(input_file);

//        writer.open("yolox.mp4", codec, 25, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            test_yolox(interpreter, bgr, result);
//            writer << result;
            cv::imshow("YoloX", result);
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