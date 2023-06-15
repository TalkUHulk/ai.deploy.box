//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBYOLOV7_HPP
#define AIDB_AIDBYOLOV7_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"

namespace py = pybind11;

class AiDBYoloV7{
    AIDB::Interpreter *detect_ins{};
    bool _grid = false;
public:
    AiDBYoloV7() = default;
    ~AiDBYoloV7() {
        if(detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(detect_ins);
            detect_ins = nullptr;
        }
    }
    AiDBYoloV7(const std::string& model, const std::string& backend, const std::string& config_zoo, bool grid=false){
        detect_ins = AIDB::Interpreter::createInstance(model, backend, config_zoo);
        _grid = grid;
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

        if(_grid){
            AIDB::Utility::yolov7_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.25, detect_ins->scale_h());
        } else{
            AIDB::Utility::yolov7_post_process(outputs, outputs_shape, object, 0.45, 0.25, detect_ins->scale_h());
        }

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

#endif //AIDB_AIDBYOLOV7_HPP
