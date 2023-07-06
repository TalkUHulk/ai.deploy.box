//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBFACE3DDFANODE_HPP
#define AIDB_AIDBFACE3DDFANODE_HPP


#include <Interpreter.h>
#include "AiDBNode.hpp"
namespace AiDBServer {
    class AiDBFace3DDFANode : public AiDBNode {
    public:
        AiDBFace3DDFANode() = default;

        ~AiDBFace3DDFANode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    };
}
#endif //AIDB_AIDBFACE3DDFANODE_HPP
