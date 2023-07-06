//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBYOLOXNODE_HPP
#define AIDB_AIDBYOLOXNODE_HPP

#include <Interpreter.h>
#include "AiDBNode.hpp"
namespace AiDBServer {
    class AiDBYoloXNode : public AiDBNode {
    public:
        AiDBYoloXNode() = default;

        ~AiDBYoloXNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    };
}
#endif //AIDB_AIDBYOLOXNODE_HPP
