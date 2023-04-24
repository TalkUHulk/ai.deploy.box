//
// Created by TalkUHulk on 2023/3/27.
//

#ifndef AIDEPLOYBOX_PFPLD_H
#define AIDEPLOYBOX_PFPLD_H

#include "models/pfpld.mem.h"
#include "models/pfpld.id.h"
#include <iostream>
#include <string>
#include "utility/Utility.h"
#include "Interpreter.h"
#include <simpleocv.h>

#define PI 3.1415926
class PFPLD{
public:
    PFPLD() = default;
    ~PFPLD(){
        if(_ins){
            AIDB::Interpreter::releaseInstance(_ins);
        }
    }

    int init(){
        _ins = AIDB::Interpreter::createInstance(pfpld_kps98_simplify_param_bin, pfpld_kps98_simplify_bin, _config);
        return 0;
    }

    int detect(const cv::Mat& rgba, std::vector<std::shared_ptr<AIDB::FaceMeta>> &face_metas){

        int a = 0;

        for(auto &face_meta: face_metas){

            std::vector<std::vector<float>> outputs;
            std::vector<std::vector<int>> outputs_shape;

            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, rgba.cols, rgba.rows, 1.28, 0.14);

            auto roi = cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height());
//            std::cout << "roi:" << face_meta_roi->x1<< ";" << face_meta_roi->y1<< ";" << face_meta_roi->width()<< ";" << face_meta_roi->height()<< ";" << std::endl;
            _ins->set_roi(roi);

            ncnn::Mat blob = *_ins << rgba;

            _ins->forward(blob.data, blob.w, blob.h, blob.c, outputs, outputs_shape);
            AIDB::Utility::pfpld_post_process(outputs, face_meta_roi, face_meta, 98);

            a += 1;
        }

        return 0;
    }

    int draw(cv::Mat& rgba, const std::vector<std::shared_ptr<AIDB::FaceMeta>>& face_metas){
        for(auto &face_meta: face_metas){
            int baseLine = 0;
            char text[10];
            sprintf(text, "%.2f", face_meta->score);

            cv::Size label_size = cv::getTextSize(text,
                                                  cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int y = face_meta->y1 - label_size.height;
            int x = face_meta->x1;

            cv::rectangle(rgba, cv::Rect(cv::Point(x, y - baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(rgba, text, cv::Point(x, y + label_size.height - baseLine),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255));

            cv::rectangle(rgba, cv::Point(int(face_meta->x1), int(face_meta->y1)),
                          cv::Point(int(face_meta->x2), int(face_meta->y2)),
                          cv::Scalar(255, 0, 0), 2);

            for(int n = 0; n < face_meta->kps.size(); n++){
                cv::circle(rgba, cv::Point(int(face_meta->kps[2 * n]), int(face_meta->kps[2 * n + 1])), 3, cv::Scalar(0, 0, 255), -1);
            }

        }
        return 0;
    }


private:
    AIDB::Interpreter* _ins;

    std::string _config = "name: \"pfpld\"\n"
                          "backend: \"wasm\"\n"
                          "num_thread: 2\n"
                          "device: \"CPU\"\n"
                          "PreProcess:\n"
                          "    shape: &shape\n"
                          "            width: 112\n"
                          "            height: 112\n"
                          "            channel: 3\n"
                          "            batch: 1\n"
                          "    keep_ratio: true\n"
                          "    mean:\n"
                          "        - 0\n"
                          "        - 0\n"
                          "        - 0\n"
                          "    var:\n"
                          "        - 255.0\n"
                          "        - 255.0\n"
                          "        - 255.0\n"
                          "    border_constant:\n"
                          "        - 0.0\n"
                          "        - 0.0\n"
                          "        - 0.0\n"
                          "    originimageformat: \"RGBA\"\n"
                          "    imageformat: \"RGB\"\n"
                          "    inputformat: &format \"NCHW\"\n"
                          "input_node1: &in_node1\n"
                          "    input_name: \"0\"\n"
                          "    format: *format\n"
                          "    shape: *shape\n"
                          "input_nodes:\n"
                          "    - *in_node1\n"
                          "output_nodes:\n"
                          "    - \"128\"\n"
                          "    - \"115\"";
};

#endif //AIDEPLOYBOX_PFPLD_H
