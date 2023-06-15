//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBPPOCR_HPP
#define AIDB_AIDBPPOCR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"
#include <opencv2/opencv.hpp>

namespace py = pybind11;

class AiDBPPOcr{
    AIDB::Interpreter *detect_ins{};
    AIDB::Interpreter *cls_ins{};
    AIDB::Interpreter *rec_ins{};
    AIDB::Utility::PPOCR post_process = AIDB::Utility::PPOCR();
public:
    AiDBPPOcr() = delete;
    ~AiDBPPOcr() {
        if(detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(detect_ins);
            detect_ins = nullptr;
        }
        if(cls_ins != nullptr){
            AIDB::Interpreter::releaseInstance(cls_ins);
            cls_ins = nullptr;
        }
        if(rec_ins != nullptr){
            AIDB::Interpreter::releaseInstance(rec_ins);
            rec_ins = nullptr;
        }
        
    }
    AiDBPPOcr(const std::string& model1, const std::string& backend1,
              const std::string& model2, const std::string& backend2,
              const std::string& model3, const std::string& backend3,
              const std::string& config_zoo){
        detect_ins = AIDB::Interpreter::createInstance(model1, backend1, config_zoo);
        cls_ins = AIDB::Interpreter::createInstance(model2, backend2, config_zoo);
        rec_ins = AIDB::Interpreter::createInstance(model3, backend3, config_zoo);
    }

    void forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height, py::dict &result) {
        py::buffer_info buf = frame_array.request();
        auto* frame_ptr = (uint8_t*)buf.ptr;

        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_ptr);

        cv::Mat blob = *detect_ins << frame;
        
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        detect_ins->forward((float*)blob.data, detect_ins->width(), detect_ins->height(), detect_ins->channel(), outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::OcrMeta>> ocr_results;

        post_process.dbnet_post_process(outputs[0], outputs_shape[0], ocr_results, detect_ins->scale_h(), detect_ins->scale_w(), frame);

        py::list ocr_meta;
        for (auto& ocr_result : ocr_results) {

            cv::Mat crop_img;
            AIDB::Utility::PPOCR::GetRotateCropImage(frame, crop_img, ocr_result);

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

            std::vector<std::vector<int>> box;
            for (auto & m : ocr_result->box) {
                box.push_back({m._x, m._y});
            }

            py::dict ocr_meta_dict;
            py::list det = py::cast(box);

            ocr_meta_dict["label"] = ocr_result->label;
            ocr_meta_dict["box"] = box;
            ocr_meta_dict["conf"] = ocr_result->conf;
            ocr_meta.append(ocr_meta_dict);

        }

        result["ocr"] = ocr_meta;
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBPPOCR_HPP
