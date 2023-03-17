//
// Created by TalkUHulk on 2022/11/2.
//

#ifndef AIENGINE_TNNPARAMETER_HPP
#define AIENGINE_TNNPARAMETER_HPP
#include "core/Parameter.hpp"

namespace AIDB {

    class TNNParameter: public Parameter {
        EngineID _engine_id = TNN;

    public:
        ~TNNParameter() override= default;
        explicit TNNParameter(const YAML::Node& node);
    };
}

#endif //AIENGINE_TNNPARAMETER_HPP
