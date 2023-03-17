//
// Created by TalkUHulk on 2022/12/28.
//

#ifndef AIENGINE_PPLITEPARAMETER_HPP
#define AIENGINE_PPLITEPARAMETER_HPP
#include "core/Parameter.hpp"

namespace AIDB {

    class PPLiteParameter: public Parameter {
        EngineID _engine_id = PADDLE_LITE;

    public:
        ~PPLiteParameter() override= default;
        explicit PPLiteParameter(const YAML::Node& node);
    };
}

#endif //AIENGINE_PPLitePARAMETER_HPP
