//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_AIDBPPDBNETNODE_HPP
#define AIDB_AIDBPPDBNETNODE_HPP


#include <Interpreter.h>
#include "AiDBNode.hpp"
#include <utility/Utility.h>

namespace AiDBServer {
    class AiDBPPDBNetNode : public AiDBNode {
    public:
        AiDBPPDBNetNode() = default;

        ~AiDBPPDBNetNode() override = default;

        int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) override;

    private:
        AIDB::Utility::PPOCR _post_process = AIDB::Utility::PPOCR();
    };
}
#endif //AIDB_AIDBPPDBNETNODE_HPP
