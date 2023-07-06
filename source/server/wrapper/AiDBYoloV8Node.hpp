//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBYOLOV8NODE_HPP
#define AIDB_AIDBYOLOV8NODE_HPP

#include <Interpreter.h>
#include "AiDBNode.hpp"
namespace AiDBServer {
    class AiDBYoloV8Node : public AiDBNode {
    public:
        AiDBYoloV8Node() = default;

        ~AiDBYoloV8Node() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;
    };
}
#endif //AIDB_AIDBYOLOV8NODE_HPP
