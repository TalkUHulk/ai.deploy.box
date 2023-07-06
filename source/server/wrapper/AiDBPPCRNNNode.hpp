//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBPPCRNNNODE_HPP
#define AIDB_AIDBPPCRNNNODE_HPP

#include <Interpreter.h>
#include "AiDBNode.hpp"
#include <utility/Utility.h>

namespace AiDBServer {
    class AiDBPPCRNNNode : public AiDBNode {
    public:
        AiDBPPCRNNNode() = default;

        ~AiDBPPCRNNNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    private:
        float _cls_thresh = 0.9;
        AIDB::Utility::PPOCR _post_process = AIDB::Utility::PPOCR();
    };
}
#endif //AIDB_AIDBPPCRNNNODE_HPP
