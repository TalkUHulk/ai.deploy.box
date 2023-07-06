//
// Created by TalkUHulk on 2023/7/4.
//

#ifndef AIDB_AIDBANIMEGANNODE_HPP
#define AIDB_AIDBANIMEGANNODE_HPP


#include <Interpreter.h>
#include "AiDBNode.hpp"

namespace AiDBServer {
    class AiDBAnimeGanNode : public AiDBNode {
    public:
        AiDBAnimeGanNode() = default;

        ~AiDBAnimeGanNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    };
}
#endif //AIDB_AIDBANIMEGANNODE_HPP
