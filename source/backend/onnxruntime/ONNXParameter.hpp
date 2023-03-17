//
// Created by TalkUHulk on 2022/10/18.
//

#ifndef AIENGINE_ONNXPARAMETER_HPP
#define AIENGINE_ONNXPARAMETER_HPP

#include "core/Parameter.hpp"

namespace AIDB {

    class ONNXParameter: public Parameter {
        EngineID _engine_id = ONNX;
    public:
        ~ONNXParameter() override =default;
        explicit ONNXParameter(const YAML::Node& node);
    };
}

#endif //AIENGINE_ONNXPARAMETER_HPP
