//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBPPCLSNODE_HPP
#define AIDB_AIDBPPCLSNODE_HPP

#include <Interpreter.h>
#include "AiDBNode.hpp"
#include <utility/Utility.h>

namespace AiDBServer {
    class AiDBPPCLsNode : public AiDBNode {
    public:
        AiDBPPCLsNode() = default;

        ~AiDBPPCLsNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;
    };
}
#endif //AIDB_AIDBPPCLSNODE_HPP
