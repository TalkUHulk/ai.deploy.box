//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBMOBILEVITNODE_HPP
#define AIDB_AIDBMOBILEVITNODE_HPP


#include <Interpreter.h>
#include "AiDBNode.hpp"

namespace AiDBServer {
    class AiDBMobileVitNode : public AiDBNode {
    public:
        AiDBMobileVitNode() = default;

        ~AiDBMobileVitNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    private:
        int _topK = 3;
    };
}
#endif //AIDB_AIDBMOBILEVITNODE_HPP
