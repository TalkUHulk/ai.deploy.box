//
// Created by TalkUHulk on 2023/6/29.
//

#ifndef AIDB_AIDB_FLOW_HPP
#define AIDB_AIDB_FLOW_HPP

#include <map>
#include <string>
#include "Interpreter.h"
#include "utility/Utility.h"
#include "nlohmann/json.hpp"
#include "wrapper/AiDBWrapper.hpp"

using json = nlohmann::json;

namespace AiDBServer {
    class AiDBFlow {
    public:
        AiDBFlow() = delete;

        explicit AiDBFlow(const json &input);

        ~AiDBFlow();

        int Forward(const unsigned char *frame, int frame_width, int frame_height, json &result);

        static std::string model_parsing(const std::string &model);

    private:
        std::vector<std::string> _task_queue;
        std::map<std::string, AiDBBaseNode *> _task_node;
        std::map<std::string, std::string> _task_backend;
        std::string _flow_uuid;
    };

}
#endif //AIDB_AIDB_FLOW_HPP
