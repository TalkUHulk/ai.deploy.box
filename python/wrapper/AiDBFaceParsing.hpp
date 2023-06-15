//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBFACEPARSING_HPP
#define AIDB_AIDBFACEPARSING_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"
#include "utility/face_align.h"

namespace py = pybind11;

class AiDBFaceParsing{
    AIDB::Interpreter *face_detect_ins{};
    AIDB::Interpreter *face_parsing_ins{};
public:
    AiDBFaceParsing() = delete;
    ~AiDBFaceParsing() {
        if(face_detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(face_detect_ins);
            face_detect_ins = nullptr;
        }
        if(face_parsing_ins != nullptr){
            AIDB::Interpreter::releaseInstance(face_parsing_ins);
            face_parsing_ins = nullptr;
        }
    }
    AiDBFaceParsing(const std::string& model1, const std::string& backend1,
                     const std::string& model2, const std::string& backend2, const std::string& config_zoo){
        face_detect_ins = AIDB::Interpreter::createInstance(model1, backend1, config_zoo);
        face_parsing_ins = AIDB::Interpreter::createInstance(model2, backend2, config_zoo);
    }

    void forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height, py::dict &result) {
        py::buffer_info buf = frame_array.request();
        auto* frame_ptr = (uint8_t*)buf.ptr;

        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_ptr);
        cv::Mat blob = *face_detect_ins << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        face_detect_ins->forward((float*)blob.data, face_detect_ins->width(), face_detect_ins->height(), face_detect_ins->channel(), outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;

        assert(face_detect_input->scale_h() == face_detect_input->scale_w());
        AIDB::Utility::scrfd_post_process(outputs,
                                          face_metas,
                                          face_detect_ins->width(),
                                          face_detect_ins->height(),
                                          face_detect_ins->scale_h());

        py::list faces_meta;

        for(auto &face_meta: face_metas){

            cv::Mat align;
            AIDB::faceAlign(frame, align, face_meta, face_parsing_ins->width(), "ffhq");

            blob = *face_parsing_ins << align;

            cv::Mat src_image;
            *face_parsing_ins >> src_image;

            outputs.clear();
            outputs_shape.clear();

            face_parsing_ins->forward((float*)blob.data, face_parsing_ins->width(), face_parsing_ins->height(), face_parsing_ins->channel(), outputs, outputs_shape);
            cv::Mat G = src_image.clone();
            AIDB::Utility::bisenet_post_process(src_image, G, outputs[0], outputs_shape[0]);

            py::dict face_meta_dict;

            std::vector<float> bbox{face_meta->x1,
                                    face_meta->y1,
                                    face_meta->x2,
                                    face_meta->y2};

            py::list det = py::cast(bbox);
            face_meta_dict["G"] = py::array_t<uint8_t>({ G.rows,G.cols, 3 }, G.data);
            face_meta_dict["bbox"] = det;
            face_meta_dict["conf"] = face_meta->score;
            faces_meta.append(face_meta_dict);
        }

        result["face"] = faces_meta;
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBFACEPARSING_HPP
