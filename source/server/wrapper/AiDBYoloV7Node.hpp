//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBYOLOV7NODE_HPP
#define AIDB_AIDBYOLOV7NODE_HPP

#include <Interpreter.h>
#include "AiDBNode.hpp"
namespace AiDBServer {
    class AiDBYoloV7Node : public AiDBNode {
    public:
        AiDBYoloV7Node() = default;

        ~AiDBYoloV7Node() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;
    };

}
#endif //AIDB_AIDBYOLOV7NODE_HPP
