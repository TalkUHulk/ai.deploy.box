//
// Created by TalkUHulk on 2023/6/29.
//

#ifndef AIDB_AIDB_SERVER_H
#define AIDB_AIDB_SERVER_H

#include <map>
#include <string>
#include "Interpreter.h"
#include "utility/Utility.h"
#include "nlohmann/json.hpp"
#include "aidb_flow.hpp"
#include "reflect.hpp"

using json = nlohmann::json;

namespace AiDBServer {
//定义创建对象的函数指针
    typedef void *(*createAiDBNode)();

    class AiDBServer {
    public:
        AiDBServer();

//    explicit AiDBServer(const char* parameter);
        ~AiDBServer();

        int unRegister(const char *parameter);

        int Register(const char *flow_uuid);

        int Forward(const char *binary_image, const char *flow_uuid, char *binary_result, int size_in, int *size_out);

    private:
        std::map<std::string, AiDBFlow *> _task_flow;
        std::string _zoo = "./config";
    };
}
#endif //AIDB_AIDB_SERVER_H
