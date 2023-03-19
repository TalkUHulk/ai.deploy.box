//
// Created by TalkUHulk on 2022/10/19.
//

#ifndef AIENGINE_COMMON_HPP
#define AIENGINE_COMMON_HPP

#include "AIDBDefine.h"
#include <string>
namespace AIDB {

    inline Device deviceType(const std::string& _d){
        if(_d == "GPU" || _d == "gpu" || _d == "Gpu")
            return GPU;
        else
            return CPU;
    };

    inline EngineID engineType(const std::string& _d){
        if(_d == "Onnx" || _d == "ONNX" || _d == "onnx")
            return ONNX;
        else if(_d == "MNN" || _d == "mnn")
            return MNN;
        else if(_d == "NCNN" || _d == "ncnn")
            return NCNN;
        else if(_d == "TNN" || _d == "tnn")
            return TNN;
        else if(_d == "OPENVINO" || _d == "openvino" || _d == "Openvino" || _d == "OpenVINO")
            return OPENVINO;
        else if(_d == "PaddleLite" || _d == "Paddle" || _d == "paddlelite" || _d == "paddle")
            return PADDLE_LITE;
        else if(_d == "TensorRT" || _d == "tensorrt" || _d == "TENSORRT" || _d == "Tensorrt")
            return TRT;
        else
            return IDLE;
    }
}
#endif //AIENGINE_COMMON_HPP
