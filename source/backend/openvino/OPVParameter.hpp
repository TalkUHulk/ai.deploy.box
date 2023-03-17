//
// Created by TalkUHulk on 2022/11/03.
//

#ifndef AIENGINE_OPVPARAMETER_HPP
#define AIENGINE_OPVPARAMETER_HPP
#include "core/Parameter.hpp"

namespace AIDB {

    class OPVParameter: public Parameter {
        EngineID _engine_id = OPENVINO;

    public:
        ~OPVParameter() override= default;
        explicit OPVParameter(const YAML::Node& node);
    };
}

#endif //AIENGINE_OPVPARAMETER_HPP
