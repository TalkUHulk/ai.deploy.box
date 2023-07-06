//
// Created by TalkUHulk on 2023/6/20.
//
#include "aidb_c_api.h"
#include "Interpreter.h"
#include "utility/Utility.h"
#include "nlohmann/json.hpp"
#include "server/aidb_server.hpp"
#include "utility/log.h"
using json = nlohmann::json;


AiDB AiDBCreate(){
    AIDB::aidb_log_init(AIDB_DEBUG, "debug");
    auto aidb_ins = new AiDBServer::AiDBServer();
    return (void*)aidb_ins;
}

void AiDBFree(AiDB ins){
    if(nullptr != ins){
        auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
        delete aidb_ins;
        aidb_ins = nullptr;
    }
}

int AiDBRegister(AiDB ins,const char* parameter){
    if(nullptr == ins){
        std::cout << "AiDBRegister null \n";
        return -1;
    }
    auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
    return aidb_ins->Register(parameter);
}
int AiDBUnRegister(AiDB ins, const char* flow_uuid){
    if(nullptr == ins){
        std::cout << "AiDBUnRegister null \n";
        return -1;
    }
    auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
    return aidb_ins->unRegister(flow_uuid);
}

int AiDBForward(AiDB ins, const char* flow_uuid, const char* binary_image, char* binary_result, int size_in, int* size_out){
    if(nullptr == ins){
        std::cout << "AiDBForward null \n";
        return -1;
    }
    auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
    return aidb_ins->Forward(binary_image, flow_uuid, binary_result, size_in, size_out);
}


