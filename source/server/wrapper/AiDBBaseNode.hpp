//
// Created by TalkUHulk on 2023/6/29.
//

#ifndef AIDB_AIDBBASENODE_HPP
#define AIDB_AIDBBASENODE_HPP

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace AiDBServer {
    class AiDBBaseNode {
    public:
        AiDBBaseNode() = default;

        virtual ~AiDBBaseNode() = default;

        virtual int init(const std::string &model, const std::string &backend, const std::string &config_zoo) = 0;

        virtual int forward(const unsigned char *frame, int frame_width, int frame_height, json &result) = 0;

    };
}
#endif //AIDB_AIDBBASENODE_HPP
