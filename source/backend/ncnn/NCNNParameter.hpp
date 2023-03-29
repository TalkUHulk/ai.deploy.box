//
// Created by TalkUHulk on 2022/10/25.
//

#ifndef AIENGINE_NCNNPARAMETER_HPP
#define AIENGINE_NCNNPARAMETER_HPP
#include "core/Parameter.hpp"

namespace AIDB {

    class NCNNParameter: public Parameter {
        EngineID _engine_id = NCNN;
    public:
        ~NCNNParameter() override= default;
        explicit NCNNParameter(const YAML::Node& node);
        explicit NCNNParameter(const std::string& node);
    };


}

#endif //AIENGINE_NCNNPARAMETER_HPP
