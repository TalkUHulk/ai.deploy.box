//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBFACEPARSINGNODE_HPP
#define AIDB_AIDBFACEPARSINGNODE_HPP


#include <Interpreter.h>
#include "AiDBNode.hpp"

namespace AiDBServer {
    class AiDBFaceParsingNode : public AiDBNode {
    public:
        AiDBFaceParsingNode() = default;

        ~AiDBFaceParsingNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    };
}
#endif //AIDB_AIDBFACEPARSINGNODE_HPP
