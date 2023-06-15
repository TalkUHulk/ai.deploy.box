//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBANIMEGAN_HPP
#define AIDB_AIDBANIMEGAN_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"

namespace py = pybind11;

class AiDBAnimeGan{
    AIDB::Interpreter *gan_ins{};
public:
    AiDBAnimeGan() = default;
    ~AiDBAnimeGan() {
        if(gan_ins != nullptr){
            AIDB::Interpreter::releaseInstance(gan_ins);
            gan_ins = nullptr;
        }
    }
    AiDBAnimeGan(const std::string& model, const std::string& backend, const std::string& config_zoo){
        gan_ins = AIDB::Interpreter::createInstance(model, backend, config_zoo);
    }

    void forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height, py::dict &result) {
        py::buffer_info buf = frame_array.request();
        auto* frame_ptr = (uint8_t*)buf.ptr;

        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_ptr);
        cv::Mat blob = *gan_ins << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        gan_ins->forward((float*)blob.data, gan_ins->width(), gan_ins->height(), gan_ins->channel(), outputs, outputs_shape);

        auto G = frame.clone();
        AIDB::Utility::animated_gan_post_process(outputs[0], outputs_shape[0], G);

        result["G"] = py::array_t<uint8_t>({ G.rows,G.cols, 3 }, G.data);
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBANIMEGAN_HPP
