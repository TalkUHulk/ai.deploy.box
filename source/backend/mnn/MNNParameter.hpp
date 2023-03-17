//
// Created by TalkUHulk on 2022/10/18.
//

#ifndef AIENGINE_MNNPARAMETER_HPP
#define AIENGINE_MNNPARAMETER_HPP
#include "core/Parameter.hpp"
#include <MNN/Interpreter.hpp>

namespace AIDB {

    class MNNParameter: public Parameter {
        EngineID _engine_id = MNN;
    public:
        ~MNNParameter() override= default;
        explicit MNNParameter(const YAML::Node& node);
    };
}

#endif //AIENGINE_MNNPARAMETER_HPP
