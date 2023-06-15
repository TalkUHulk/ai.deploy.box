//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBMOBILEVIT_HPP
#define AIDB_AIDBMOBILEVIT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"

namespace py = pybind11;

class AiDBMobileVit{
    AIDB::Interpreter *cls_ins{};
    int topK = 3;
public:
    AiDBMobileVit() = default;
    ~AiDBMobileVit() {
        if(cls_ins != nullptr){
            AIDB::Interpreter::releaseInstance(cls_ins);
            cls_ins = nullptr;
        }
    }
    AiDBMobileVit(const std::string& model, const std::string& backend, const std::string& config_zoo){
        cls_ins = AIDB::Interpreter::createInstance(model, backend, config_zoo);
    }

    void forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height, py::dict &result) {
        py::buffer_info buf = frame_array.request();
        auto* frame_ptr = (uint8_t*)buf.ptr;

        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_ptr);
        cv::Mat blob = *cls_ins << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        cls_ins->forward((float*)blob.data, cls_ins->width(), cls_ins->height(), cls_ins->channel(), outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::ClsMeta>> tmp;
        tmp.resize(outputs_shape[0][1]);
        for(int i = 0; i < outputs_shape[0][1]; i++){
            std::shared_ptr<AIDB::ClsMeta> meta = std::make_shared<AIDB::ClsMeta>();
            meta->conf = outputs[0][i];
            meta->label = i;
            tmp[i] = meta;
        }
        std::sort(tmp.begin(), tmp.end(), [](const std::shared_ptr<AIDB::ClsMeta> &a, const std::shared_ptr<AIDB::ClsMeta> &b){ return a->conf > b->conf;});


        py::list cls_meta;
        for(int i = 0; i < topK; i++){

            py::dict cls_meta_dict;
            cls_meta_dict["label"] = tmp[i]->label;
            cls_meta_dict["conf"] = tmp[i]->conf;
            cls_meta.append(cls_meta_dict);

        }

        result["cls"] = cls_meta;
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBMOBILEVIT_HPP
