//
// Created by TalkUHulk on 2023/3/31.
//

#ifndef AIDEPLOYBOX_MOVENET_H
#define AIDEPLOYBOX_MOVENET_H

#include "models/movenet.mem.h"
#include <iostream>
#include <string>
#include "utility/Utility.h"
#include "Interpreter.h"
#include <simpleocv.h>
#include "extra.h"


class MoveNet{
public:
    MoveNet() = default;
    ~MoveNet(){
        if(_ins){
            AIDB::Interpreter::releaseInstance(_ins);
        }
    }

    int init(){
        _ins = AIDB::Interpreter::createInstance(movenet_simplify_param_bin, movenet_simplify_bin, _config);
        return 0;
    }

    int detect(const cv::Mat& rgba, cv::Mat& result){

        ncnn::Mat blob = *_ins << rgba;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        _ins->forward(blob.data, blob.w, blob.h, blob.c, outputs, outputs_shape);

        AIDB::Utility::movenet_post_process(rgba, result, outputs, outputs_shape);
        return 0;
    }


private:
    AIDB::Interpreter* _ins;
    std::string _config = "name: \"MoveNet\"\n"
                          "backend: \"wasm\"\n"
                          "num_thread: 2\n"
                          "device: \"CPU\"\n"
                          "PreProcess:\n"
                          "    shape: &shape\n"
                          "            width: 192\n"
                          "            height: 192\n"
                          "            channel: 3\n"
                          "            batch: 1\n"
                          "    keep_ratio: false\n"
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
                          "    - \"155\"\n"
                          "    - \"160\"\n"
                          "    - \"164\"\n"
                          "    - \"168\"";
};

#endif //AIDEPLOYBOX_MoveNet_H
