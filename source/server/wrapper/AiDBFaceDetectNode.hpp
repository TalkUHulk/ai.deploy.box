//
// Created by TalkUHulk on 2023/6/29.
//

#ifndef AIDB_AIDBFACEDETECTNODE_HPP
#define AIDB_AIDBFACEDETECTNODE_HPP

#include <Interpreter.h>
#include "AiDBNode.hpp"
namespace AiDBServer {
    class AiDBFaceDetectNode : public AiDBNode {
    public:
        AiDBFaceDetectNode() = default;

        ~AiDBFaceDetectNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;
    };
}
#endif //AIDB_AIDBFACEDETECTNODE_HPP
