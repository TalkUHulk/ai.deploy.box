//
// Created by TalkUHulk on 2023/4/7.
//

#ifndef AIDEPLOYBOX_TDDFA_H
#define AIDEPLOYBOX_TDDFA_H

#include "models/tddfa_v2_mb05_bfm_head_base.mem.h"
#include "models/tddfa_v2_mb05_bfm_head_dense.mem.h"
#include "td_obj.h"
#include <iostream>
#include <string>
#include "utility/Utility.h"
#include "Interpreter.h"
#include <simpleocv.h>


class TDDFA{
public:
    TDDFA() = default;
    ~TDDFA(){
        if(_ins){
            AIDB::Interpreter::releaseInstance(_ins);
        }
    }

    int init(bool dense){
        _dense = dense;
        if(_dense){
            _ins = AIDB::Interpreter::createInstance(tddfa_v2_mb05_bfm_head_dense_simplify_param_bin, tddfa_v2_mb05_bfm_head_dense_simplify_bin, _config);
        }
        else{
            _ins = AIDB::Interpreter::createInstance(tddfa_v2_mb05_bfm_head_base_simplify_param_bin, tddfa_v2_mb05_bfm_head_base_simplify_bin, _config);
        }

        return 0;
    }

    int detect(const cv::Mat& rgba, std::vector<std::shared_ptr<AIDB::FaceMeta>> &face_metas, std::vector<float> &vertices, std::vector<float> &pose, std::vector<float> &sRt){

        for(auto &face_meta: face_metas){

            std::vector<std::vector<float>> outputs;
            std::vector<std::vector<int>> outputs_shape;

            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, rgba.cols, rgba.rows, 1.58, 0.14);

            auto roi = cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height());
//            std::cout << "roi:" << face_meta_roi->x1<< ";" << face_meta_roi->y1<< ";" << face_meta_roi->width()<< ";" << face_meta_roi->height()<< ";" << std::endl;
            _ins->set_roi(roi);
            ncnn::Mat blob = *_ins << rgba;

            _ins->forward(blob.data, blob.w, blob.h, blob.c, outputs, outputs_shape);

            AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

        }

        return 0;
    }

    int draw(cv::Mat& rgba, const std::vector<std::shared_ptr<AIDB::FaceMeta>>& face_metas,
             const std::vector<float> &vertices, const std::vector<float> &pose, const std::vector<float> &sRt){

        for(auto &face_meta: face_metas){
//            int baseLine = 0;
//            char text[10];
//            sprintf(text, "%.2f", face_meta->score);
//
//            cv::Size label_size = cv::getTextSize(text,
//                                                  cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//
//            int y = face_meta->y1 - label_size.height;
//            int x = face_meta->x1;
//
//            cv::rectangle(rgba, cv::Rect(cv::Point(x, y - baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
//                          cv::Scalar(255, 255, 255), -1);
//
//            cv::putText(rgba, text, cv::Point(x, y + label_size.height - baseLine),
//                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255));
//
//            cv::rectangle(rgba, cv::Point(int(face_meta->x1), int(face_meta->y1)),
//                          cv::Point(int(face_meta->x2), int(face_meta->y2)),
//                          cv::Scalar(255, 0, 0), 2);

            if(_dense){
                AIDB::Utility::TddfaUtility::tddfa_rasterize(rgba, vertices, hulk_obj, 1, false);
//                for(int i = 0; i < vertices.size() / 3; i+=6){
//                    cv::circle(rgba, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(255, 0, 0, 255), -1);
//                }
            }else{
                for(int i = 0; i < vertices.size() / 3; i++){
                    cv::circle(rgba, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(0, 0, 255, 255), -1);
                }
//                AIDB::Utility::TddfaUtility::plot_pose_box(result, sRt, vertices, 68);
            }

        }
        return 0;
    }


private:
    bool _dense = false;
    AIDB::Interpreter* _ins;

    std::string _config = "name: \"TDDFA\"\n"
                          "backend: \"wasm\"\n"
                          "num_thread: 2\n"
                          "device: \"CPU\"\n"
                          "PreProcess:\n"
                          "    shape: &shape\n"
                          "            width: 120\n"
                          "            height: 120\n"
                          "            channel: 3\n"
                          "            batch: 1\n"
                          "    keep_ratio: true\n"
                          "    mean:\n"
                          "        - 127.5\n"
                          "        - 127.5\n"
                          "        - 127.5\n"
                          "    var:\n"
                          "        - 128.0\n"
                          "        - 128.0\n"
                          "        - 128.0\n"
                          "    border_constant:\n"
                          "        - 0.0\n"
                          "        - 0.0\n"
                          "        - 0.0\n"
                          "    originimageformat: \"RGBA\"\n"
                          "    imageformat: \"BGR\"\n"
                          "    inputformat: &format \"NCHW\"\n"
                          "input_node1: &in_node1\n"
                          "    input_name: \"0\"\n"
                          "    format: *format\n"
                          "    shape: *shape\n"
                          "input_nodes:\n"
                          "    - *in_node1\n"
                          "output_nodes:\n"
                          "    - \"65\"\n"
                          "    - \"70\"";
};

#endif //AIDEPLOYBOX_TDDFA_H
