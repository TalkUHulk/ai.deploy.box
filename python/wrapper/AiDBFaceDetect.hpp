//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBFACEDETECT_HPP
#define AIDB_AIDBFACEDETECT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"

namespace py = pybind11;

class AiDBFaceDetect{
    AIDB::Interpreter *face_detect_ins{};
public:
    AiDBFaceDetect() = default;
    ~AiDBFaceDetect() {
        if(face_detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(face_detect_ins);
            face_detect_ins = nullptr;
        }
    }
    AiDBFaceDetect(const std::string& model, const std::string& backend, const std::string& config_zoo){
        face_detect_ins = AIDB::Interpreter::createInstance(model, backend, config_zoo);
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
            py::dict face_meta_dict;

            std::vector<float> bbox{face_meta->x1,
                                    face_meta->y1,
                                    face_meta->x2,
                                    face_meta->y2};

            py::list det = py::cast(bbox);

            std::vector<std::vector<float>> kps;
            for(int n = 0; n < face_meta->kps.size() / 2; n++){
                kps.push_back({face_meta->kps[2 * n], face_meta->kps[2 * n + 1]});
            }
            py::list landmarks = py::cast(kps);

            face_meta_dict["bbox"] = det;
            face_meta_dict["landmarks"] = landmarks;
            face_meta_dict["conf"] = face_meta->score;
            faces_meta.append(face_meta_dict);
        }

        result["face"] = faces_meta;
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBFACEDETECT_HPP
