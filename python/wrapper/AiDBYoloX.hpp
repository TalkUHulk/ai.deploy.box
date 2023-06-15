//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBYOLOX_HPP
#define AIDB_AIDBYOLOX_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"

namespace py = pybind11;

class AiDBYoloX{
    AIDB::Interpreter *detect_ins{};
public:
    AiDBYoloX() = default;
    ~AiDBYoloX() {
        if(detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(detect_ins);
            detect_ins = nullptr;
        }
    }
    AiDBYoloX(const std::string& model, const std::string& backend, const std::string& config_zoo){
        detect_ins = AIDB::Interpreter::createInstance(model, backend, config_zoo);
    }

    void forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height, py::dict &result) {
        py::buffer_info buf = frame_array.request();
        auto* frame_ptr = (uint8_t*)buf.ptr;

        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_ptr);
        cv::Mat blob = *detect_ins << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        detect_ins->forward((float*)blob.data,
                            detect_ins->width(),
                            detect_ins->height(),
                            detect_ins->channel(),
                            outputs,
                            outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

        assert(ins->scale_h() == ins->scale_w());

        auto post_process = AIDB::Utility::YoloX(detect_ins->width(), 0.25, 0.45, {8, 16, 32});

        post_process(outputs[0], outputs_shape[0], object, frame.cols, frame.rows, detect_ins->scale_h());

        py::list objs_meta;
        for (const auto & obj : object){

            std::vector<float> bbox{obj->x1,
                                    obj->y1,
                                    obj->x2,
                                    obj->y2};

            py::list det = py::cast(bbox);
            py::dict obj_meta_dict;
            obj_meta_dict["bbox"] = det;
            obj_meta_dict["conf"] = obj->score;
            obj_meta_dict["label"] = obj->label;
            objs_meta.append(obj_meta_dict);

        }

        result["object"] = objs_meta;
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBYOLOX_HPP
