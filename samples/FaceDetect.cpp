//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include <memory>
#if __linux__
#include <fstream>
#endif

void test_detect(AIDB::Interpreter* det_ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *det_ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    det_ins->forward((float*)blob.data, det_ins->width(), det_ins->height(), det_ins->channel(), outputs, outputs_shape);

    std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;

    assert(face_detect_input->scale_h() == face_detect_input->scale_w());
    AIDB::Utility::scrfd_post_process(outputs, face_metas, det_ins->width(), det_ins->height(), det_ins->scale_h());

    result = bgr.clone();

    for(auto &face_meta: face_metas){
        int baseLine = 0;
        char text[10];
        sprintf(text, "%.2f", face_meta->score);

        cv::Size label_size = cv::getTextSize(text,
                                              cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int y = face_meta->y1 - label_size.height;
        int x = face_meta->x1;

        cv::rectangle(result, cv::Rect(cv::Point(x, y - baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(result, text, cv::Point(x, y + label_size.height - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255));

        cv::rectangle(result, cv::Point(int(face_meta->x1), int(face_meta->y1)),
                      cv::Point(int(face_meta->x2), int(face_meta->y2)),
                      cv::Scalar(255, 0, 0), 2);

        for(int n = 0; n < face_meta->kps.size() / 2; n++){
            cv::circle(result, cv::Point(int(face_meta->kps[2 * n]), int(face_meta->kps[2 * n + 1])), 3, cv::Scalar(0, 0, 255), -1);
        }

    }
}

int main(int argc, char** argv){

    auto face_detect_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == face_detect_interpreter){
        return -1;
    }

    int input_type = std::atoi(argv[3]);
    std::string input_file = argv[4];


    if(0 == input_type){
        auto bgr = cv::imread(input_file);
        cv::Mat result;
        test_detect(face_detect_interpreter, bgr, result);
        cv::imwrite("FaceDetect.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("FaceDetect", result);
            cv::waitKey();
        }
#else
        cv::imshow("FaceDetect", result);
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
//        writer.open("face_det.mp4", codec, 25, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            test_detect(face_detect_interpreter, bgr, result);
//            writer << result;
            cv::imshow("FaceDetect", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
//        writer.release();
    }


    AIDB::Interpreter::releaseInstance(face_detect_interpreter);
    return 0;
}