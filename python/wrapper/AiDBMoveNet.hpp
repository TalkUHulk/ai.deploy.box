//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBMOVENET_HPP
#define AIDB_AIDBMOVENET_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"

namespace py = pybind11;

class AiDBMoveNet{
    AIDB::Interpreter *detect_ins{};
public:
    AiDBMoveNet() = default;
    ~AiDBMoveNet() {
        if(detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(detect_ins);
            detect_ins = nullptr;
        }
    }
    AiDBMoveNet(const std::string& model, const std::string& backend, const std::string& config_zoo){
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

        std::vector<std::vector<float>> decoded_keypoints;
        AIDB::Utility::movenet_post_process(frame, outputs, outputs_shape, decoded_keypoints);

        std::vector<std::vector<float>> dkps;
        dkps.reserve(decoded_keypoints.size());
        for(auto& kps: decoded_keypoints){
            dkps.push_back({kps[0], kps[1]});
        }

        py::list landmarks = py::cast(dkps);


        result["body_landmarks"] = landmarks;
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBMOVENET_HPP
