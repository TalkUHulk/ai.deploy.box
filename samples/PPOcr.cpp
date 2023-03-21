//
// Created by TalkUHulk on 2023/2/10.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#if __linux__
#include <fstream>
#endif


auto post_process = AIDB::Utility::PPOCR();

void test_ppocr(AIDB::Interpreter* det_ins, AIDB::Interpreter* cls_ins, AIDB::Interpreter* rec_ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *det_ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    det_ins->forward((float*)blob.data, det_ins->width(), det_ins->height(), det_ins->channel(), outputs, outputs_shape);

    std::vector<std::shared_ptr<AIDB::OcrMeta>> ocr_results;

    post_process.dbnet_post_process(outputs[0], outputs_shape[0], ocr_results, det_ins->scale_h(), det_ins->scale_w(), bgr);

    result = bgr.clone();
    for (auto& ocr_result : ocr_results) {

        cv::Mat crop_img;
        AIDB::Utility::PPOCR::GetRotateCropImage(bgr, crop_img, ocr_result);

        // Cls
        outputs.clear();
        outputs_shape.clear();
        cv::Mat cls_blob = *cls_ins << crop_img;
        cls_ins->forward((float*)cls_blob.data, cls_ins->width(), cls_ins->height(), cls_ins->channel(), outputs, outputs_shape);

        post_process.cls_post_process(outputs[0], outputs_shape[0], crop_img, crop_img, ocr_result);

        // crnn
        outputs.clear();
        outputs_shape.clear();
        cv::Mat crnn_blob = *rec_ins << crop_img;
        rec_ins->forward((float*)crnn_blob.data, rec_ins->width(), rec_ins->height(), rec_ins->channel(), outputs, outputs_shape);
        post_process.crnn_post_process(outputs[0], outputs_shape[0], ocr_result);

        AIDB::Utility::PPOCR::draw_objects(result, ocr_result);
        std::cout << ocr_result->label << std::endl;

    }
}

int main(int argc, char** argv){

    if(argc != 9){
        std::cout << "xxx model backend image/camera image_file/camera id/video file\n";
        return 0;
    }

    auto dbnet_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == dbnet_interpreter){
        return -1;
    }

    auto cls_interpreter = AIDB::Interpreter::createInstance(argv[3], argv[4]);

    if(nullptr == cls_interpreter){
        return -1;
    }

    auto crnn_interpreter = AIDB::Interpreter::createInstance(argv[5], argv[6]);

    if(nullptr == crnn_interpreter){
        return -1;
    }


    int input_type = std::atoi(argv[7]);
    std::string input_file = argv[8];


    if(0 == input_type){
        auto bgr = cv::imread(input_file);
        cv::Mat result;
        test_ppocr(dbnet_interpreter, cls_interpreter, crnn_interpreter,  bgr, result);

        cv::imwrite("PPOCR.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("PPOCR", result);
            cv::waitKey();
        }
#else
        cv::imshow("PPOCR", result);
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
//        writer.open("ppocr.mp4", codec, 25, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            test_ppocr(dbnet_interpreter, cls_interpreter, crnn_interpreter,  bgr, result);
//            writer << result;
            cv::imshow("PPOCR", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
//        writer.release();
    }


    AIDB::Interpreter::releaseInstance(dbnet_interpreter);
    AIDB::Interpreter::releaseInstance(cls_interpreter);
    AIDB::Interpreter::releaseInstance(crnn_interpreter);
    return 0;
}