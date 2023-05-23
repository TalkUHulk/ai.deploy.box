//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include "utility/face_align.h"
#if __linux__
#include <fstream>
#endif

void face_parsing(AIDB::Interpreter* ins_det, AIDB::Interpreter* ins_parsing, const cv::Mat& bgr, cv::Mat &result){

    cv::Mat blob = *ins_det << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins_det->forward((float*)blob.data, ins_det->width(), ins_det->height(), ins_det->channel(), outputs, outputs_shape);

    std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;

    assert(face_detect_input->scale_h() == face_detect_input->scale_w());
    AIDB::Utility::scrfd_post_process(outputs, face_metas, ins_det->width(), ins_det->height(), ins_det->scale_h());
    result = bgr.clone();
    for(auto &face_meta: face_metas){
        cv::Mat align;
        AIDB::faceAlign(bgr, align, face_meta, ins_parsing->width(), "ffhq");

        blob = *ins_parsing << align;

        cv::Mat src_image;
        *ins_parsing >> src_image;

        outputs.clear();
        outputs_shape.clear();

        ins_parsing->forward((float*)blob.data, ins_parsing->width(), ins_parsing->height(), ins_parsing->channel(), outputs, outputs_shape);

        AIDB::Utility::bisenet_post_process(src_image, result, outputs[0], outputs_shape[0]);
    }

}


int main(int argc, char** argv){

    if(argc != 7){
        std::cout << "xxx model backend image/camera image_file/camera id/video file\n";
        return 0;
    }
    auto face_detect_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == face_detect_interpreter){
        return -1;
    }

    auto face_parsing_interpreter = AIDB::Interpreter::createInstance(argv[3], argv[4]);

    if(nullptr == face_parsing_interpreter){
        return -1;
    }

    int input_type = std::atoi(argv[5]);
    std::string input_file = argv[6];


    if(0 == input_type){
        auto bgr = cv::imread(input_file);
        cv::Mat result;
        face_parsing(face_detect_interpreter, face_parsing_interpreter, bgr, result);
        cv::imwrite("FaceParsing.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("FaceParsing", result);
            cv::waitKey();
        }
#else
        cv::imshow("FaceParsing", result);
        cv::waitKey();
#endif

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
            face_parsing(face_detect_interpreter, face_parsing_interpreter, bgr, result);
//            writer << result;

            cv::imshow("FaceParsing", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
//        writer.release();
    }


    AIDB::Interpreter::releaseInstance(face_detect_interpreter);
    AIDB::Interpreter::releaseInstance(face_parsing_interpreter);

    return 0;
}