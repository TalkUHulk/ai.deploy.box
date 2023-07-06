//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBMOVENETNODE_HPP
#define AIDB_AIDBMOVENETNODE_HPP


#include <Interpreter.h>
#include "AiDBNode.hpp"

namespace AiDBServer {
    class AiDBMoveNetNode : public AiDBNode {
    public:
        AiDBMoveNetNode() = default;

        ~AiDBMoveNetNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;
    };
}

#endif //AIDB_AIDBMOVENETNODE_HPP
