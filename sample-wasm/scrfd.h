//
// Created by TalkUHulk on 2023/3/23.
//

#ifndef AIDEPLOYBOX_SCRFD_H
#define AIDEPLOYBOX_SCRFD_H

#include "models/scrfd_500m_kps.mem.h"
#include "models/scrfd_500m_kps.id.h"
#include <iostream>
#include <string>
#include "utility/Utility.h"
#include "Interpreter.h"
#include <simpleocv.h>

class SCRFD{
public:
    SCRFD() = default;
    ~SCRFD(){
        if(_ins){
            AIDB::Interpreter::releaseInstance(_ins);
        }
    }

    int init(){
        _ins = AIDB::Interpreter::createInstance(scrfd_500m_kps_simplify_param_bin, scrfd_500m_kps_simplify_bin, _config);
        return 0;
    }

    int detect(const cv::Mat& rgba, std::vector<std::shared_ptr<AIDB::FaceMeta>> &result){

        std::cout << "######"  << std::endl;
        ncnn::Mat blob = *_ins << rgba;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        _ins->forward(blob.data, blob.w, blob.h, blob.c, outputs, outputs_shape);

        AIDB::Utility::scrfd_post_process(outputs, result, _ins->width(), _ins->height(), _ins->scale_h());
//        AIDB::Utility::scrfd_post_process(outputs, result, 640, 640, rgba.cols / 640);

        std::cout << "@@@@@@" << result.size() << std::endl;
        return 0;
    }

    int draw(cv::Mat& rgba, const std::vector<std::shared_ptr<AIDB::FaceMeta>>& face_metas){
        for(auto &face_meta: face_metas){
            int baseLine = 0;
            char text[10];
            sprintf(text, "%.2f", face_meta->score);

            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int y = face_meta->y1 - label_size.height;
            int x = face_meta->x1;

            cv::rectangle(rgba, cv::Rect(cv::Point(x, y - baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(rgba, text, cv::Point(x, y + label_size.height - baseLine),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255));

            cv::rectangle(rgba, cv::Point(int(face_meta->x1), int(face_meta->y1)),
                          cv::Point(int(face_meta->x2), int(face_meta->y2)),
                          cv::Scalar(255, 0, 0), 2);

        }
        return 0;
    }

private:
    AIDB::Interpreter* _ins;

    std::string _config = "name: \"scrfd\"\n"
                         "backend: \"wasm\"\n"
                         "num_thread: 4\n"
                         "device: \"CPU\"\n"
                         "PreProcess:\n"
                         "    shape: &shape\n"
                         "            width: 640\n"
                         "            height: 640\n"
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
                         "    imageformat: \"RGB\"\n"
                         "    inputformat: &format \"NCHW\"\n"
                         "input_node1: &in_node1\n"
                         "    input_name: \"0\"\n"
                         "    format: *format\n"
                         "    shape: *shape\n"
                         "input_nodes:\n"
                         "    - *in_node1\n"
                         "output_nodes:\n"
                         "    - \"103\"\n"
                         "    - \"124\"\n"
                         "    - \"145\"\n"
                         "    - \"105\"\n"
                         "    - \"126\"\n"
                         "    - \"147\"\n"
                         "    - \"107\"\n"
                         "    - \"128\"\n"
                         "    - \"149\"";
};
#endif //AIDEPLOYBOX_SCRFD_H

