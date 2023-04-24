//
// Created by TalkUHulk on 2023/3/31.
//

#ifndef AIDEPLOYBOX_YoloX_H
#define AIDEPLOYBOX_YoloX_H

#include "models/yolox_nano.mem.h"
#include "models/yolox_nano.id.h"
#include <iostream>
#include <string>
#include "utility/Utility.h"
#include "Interpreter.h"
#include <simpleocv.h>
#include "extra.h"


class YoloX{
public:
    YoloX() = default;
    ~YoloX(){
        if(_ins){
            AIDB::Interpreter::releaseInstance(_ins);
        }
        if(_post_process){
            delete _post_process;
            _post_process = nullptr;
        }
    }

    int init(){
        _ins = AIDB::Interpreter::createInstance(yolox_nano_param_bin, yolox_nano_bin, _config);
        _post_process = new AIDB::Utility::YoloX(_ins->height(), 0.25, 0.45, {8, 16, 32});
        return 0;
    }

    int detect(const cv::Mat& rgba, std::vector<std::shared_ptr<AIDB::ObjectMeta>> &object){

        ncnn::Mat blob = *_ins << rgba;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        _ins->forward(blob.data, blob.w, blob.h, blob.c, outputs, outputs_shape);
        _post_process->forward(outputs[0], outputs_shape[0], object, rgba.cols, rgba.rows, _ins->scale_h());
        return 0;
    }

    int draw(cv::Mat& rgba, const std::vector<std::shared_ptr<AIDB::ObjectMeta>> &objects){

        int color_index = 0;

        for (const auto & object : objects){

            const unsigned char* color = colors[color_index % 19];
            color_index++;

            cv::Scalar cc(color[0], color[1], color[2], 255);

            cv::rectangle(rgba, cv::Point(object->x1, object->y1), cv::Point(object->x2, object->y2), cc);

            char text[256];
            sprintf(text, "%s %.1f%%", coco_labels[object->label], object->score * 100);
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = object->x1;
            int y = object->y1 - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > rgba.cols)
                x = rgba.cols - label_size.width;

            cv::rectangle(rgba, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cc, -1);

            cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0, 255) : cv::Scalar(255, 255, 255, 255);

            cv::putText(rgba, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc);
        }
        return 0;
    }


private:
    AIDB::Interpreter* _ins;
    AIDB::Utility::YoloX* _post_process = nullptr;
    std::string _config = "name: \"YoloX\"\n"
                          "backend: \"wasm\"\n"
                          "num_thread: 2\n"
                          "device: \"CPU\"\n"
                          "PreProcess:\n"
                          "    shape: &shape\n"
                          "            width: 416\n"
                          "            height: 416\n"
                          "            channel: 3\n"
                          "            batch: 1\n"
                          "    keep_ratio: true\n"
                          "    mean:\n"
                          "        - 0\n"
                          "        - 0\n"
                          "        - 0\n"
                          "    var:\n"
                          "        - 1.0\n"
                          "        - 1.0\n"
                          "        - 1.0\n"
                          "    border_constant:\n"
                          "        - 114.0\n"
                          "        - 114.0\n"
                          "        - 114.0\n"
                          "    originimageformat: \"RGBA\"\n"
                          "    imageformat: \"RGB\"\n"
                          "    inputformat: &format \"NCHW\"\n"
                          "register_layer: \"YoloV5Focus\"\n"
                          "input_node1: &in_node1\n"
                          "    input_name: \"0\"\n"
                          "    format: *format\n"
                          "    shape: *shape\n"
                          "input_nodes:\n"
                          "    - *in_node1\n"
                          "output_nodes:\n"
                          "    - \"315\"";
};

#endif //AIDEPLOYBOX_YoloX_H
