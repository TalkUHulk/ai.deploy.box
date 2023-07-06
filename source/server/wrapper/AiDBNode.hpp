//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBNODE_HPP
#define AIDB_AIDBNODE_HPP

#include <Interpreter.h>
#include "AiDBBaseNode.hpp"
#include "utility/log.h"

namespace AiDBServer {
    class AiDBNode : public AiDBBaseNode {
    public:
        AiDBNode() = default;

        ~AiDBNode() override {
            AIDB::Interpreter::releaseInstance(this->_ins);
        }

        int init(const std::string &model, const std::string &backend, const std::string &config_zoo) override {
            if (nullptr != this->_ins)
                AIDB::Interpreter::releaseInstance(this->_ins);
            _ins = AIDB::Interpreter::createInstance(model, backend, config_zoo);
            return 0;
        }

    protected:
        AIDB::Interpreter *_ins{};

    };
}

#endif //AIDB_AIDBNODE_HPP
