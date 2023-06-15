//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBFACE3DDFA_HPP
#define AIDB_AIDBFACE3DDFA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include "Interpreter.h"
#include "AIDBData.h"
#include "utility/Utility.h"
#include "utility/td_obj.h"

namespace py = pybind11;

class AiDBFace3DDFA{
    AIDB::Interpreter *face_detect_ins{};
    AIDB::Interpreter *face_3dddfa_ins{};
public:
    AiDBFace3DDFA() = default;
    ~AiDBFace3DDFA() {
        if(face_detect_ins != nullptr){
            AIDB::Interpreter::releaseInstance(face_detect_ins);
            face_detect_ins = nullptr;
        }
        if(face_3dddfa_ins != nullptr){
            AIDB::Interpreter::releaseInstance(face_3dddfa_ins);
            face_3dddfa_ins = nullptr;
        }
    }
    AiDBFace3DDFA(const std::string& model1, const std::string& backend1,
                  const std::string& model2, const std::string& backend2, const std::string& config_zoo){
        face_detect_ins = AIDB::Interpreter::createInstance(model1, backend1, config_zoo);
        face_3dddfa_ins = AIDB::Interpreter::createInstance(model2, backend2, config_zoo);
    }

    void forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height, py::dict &result, bool dense=true) {
        py::buffer_info buf = frame_array.request();
        auto* frame_ptr = (uint8_t*)buf.ptr;

        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_ptr);
        auto G = frame.clone();
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

        for(auto &face_meta: face_metas){

            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, frame.cols, frame.rows, 1.58, 0.14);
            cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));

            blob = *face_3dddfa_ins << roi;
            outputs.clear();
            outputs_shape.clear();
            face_3dddfa_ins->forward((float*)blob.data, face_3dddfa_ins->width(), face_3dddfa_ins->height(), face_3dddfa_ins->channel(), outputs, outputs_shape);
            std::vector<float> vertices, pose, sRt;

            AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

            if(dense){
                AIDB::Utility::TddfaUtility::tddfa_rasterize(G, vertices, spider_man_obj, 1, true);
            }
            else{
                for(int i = 0; i < vertices.size() / 3; i++){
                    cv::circle(G, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(255, 255, 255), -1);
                }
                AIDB::Utility::TddfaUtility::plot_pose_box(G, sRt, vertices, 68);
            }

        }

        result["G"] = py::array_t<uint8_t>({ G.rows,G.cols,3 }, G.data);
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBFACE3DDFA_HPP
