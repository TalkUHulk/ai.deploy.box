//
// Created by TalkUHulk on 2023/8/1.
//

#include "wrapper.h"
#include "utility/Utility.h"
#include "server/aidb_server.hpp"
#include "utility/log.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace AIDB;

// lua_touserdata
// 如果给定索引处的值是一个完整的userdata，函数返回内存块的地址。
// 如果值是一个lightuserdata，那么就返回它表示的指针。
// 否则，返回NULL。

int AiDBCreate(lua_State *L){
    AIDB::aidb_log_init(AIDB_DEBUG, "debug");

    auto** aidb_ins = (AiDBServer::AiDBServer**)lua_newuserdata(L, sizeof(AiDBServer::AiDBServer*));
    *aidb_ins = new AiDBServer::AiDBServer();
    std::cout << "LuaAiDBCreate:" << *aidb_ins << std::endl;
    return 1;
}

int AiDBFree(lua_State *L){
    auto aidb_ins = (AiDBServer::AiDBServer**)lua_touserdata(L, 1);
//    luaL_argcheck(L, *aidb_ins != nullptr, 1, "ins is null");
    if(*aidb_ins){
        delete *aidb_ins;
        *aidb_ins = nullptr;

        lua_settop(L, 0);
    }

    return 1;
}

int AiDBRegister(lua_State *L){
    auto aidb_ins = (AiDBServer::AiDBServer**)lua_touserdata(L, 1);
    luaL_argcheck(L, *aidb_ins != nullptr, 1, "ins is null");

    size_t len;
    auto parameter = (char *)lua_tolstring(L, 2, &len);

    auto ret = (*aidb_ins)->Register(parameter);

    lua_pushnumber(L, ret);   //结果压入栈
    return 1;
}

int AiDBUnRegister(lua_State *L){
    auto aidb_ins = (AiDBServer::AiDBServer**)lua_touserdata(L, 1);
    luaL_argcheck(L, *aidb_ins != nullptr, 1, "ins is null");

    size_t len;
    auto flow_uuid = (char *)lua_tolstring(L, 2, &len);

    auto ret = (*aidb_ins)->unRegister(flow_uuid);

    lua_pushnumber(L, ret);

    return 1;
}

int AiDBForward(lua_State *L){
    int num = lua_gettop(L);

    auto aidb_ins = (AiDBServer::AiDBServer**)lua_touserdata(L, 1);
    luaL_argcheck(L, *aidb_ins != nullptr, 1, "ins is null");

    size_t len;
    auto flow_uuid = (char *)lua_tolstring(L, 2, &len);

    auto binary_image = (char *)lua_tolstring(L, 3, &len);

    int cache = (num == 4)? lua_tonumber(L, 4): 1024 * 1024;

    char* binary_result = new char[cache];
    int size_out;

    (*aidb_ins)->Forward(binary_image, flow_uuid, binary_result, cache, &size_out);

    json output = json::parse(binary_result);

    lua_newtable(L);
    lua_pushstring(L, "code");
    lua_pushnumber(L, output["error_code"]);

    lua_settable(L, -3);
    if(output.contains("tddfa")){
        lua_pushstring(L, "tddfa");
        lua_pushstring(L, output["tddfa"].get<std::string>().c_str());
        lua_settable(L, -3);
    }

    if(output.contains("anime")){
        lua_pushstring(L, "anime");
        lua_pushstring(L, output["anime"].get<std::string>().c_str());
        lua_settable(L, -3);
    }

    if(output.contains("face")){
        lua_pushstring(L, "face");
        lua_createtable(L, output["face"].size(), 0);
        int cnt = 0;
        for(auto face: output["face"]){
            lua_pushstring(L, ("face" + std::to_string(cnt++)).c_str());
            lua_newtable(L);

            lua_pushnumber(L, face["conf"]);
            lua_setfield(L, -2, "conf");

            lua_pushstring(L, "bbox");
            lua_createtable(L, 0, 4);

            lua_pushnumber(L, face["bbox"][0]);
            lua_setfield(L, -2, "x1");

            lua_pushnumber(L, face["bbox"][1]);
            lua_setfield(L, -2, "y1");

            lua_pushnumber(L, face["bbox"][2]);
            lua_setfield(L, -2, "x2");

            lua_pushnumber(L, face["bbox"][3]);
            lua_setfield(L, -2, "y2");

            if(face.contains("parsing")){
                lua_pushstring(L, ("parsing" + std::to_string(cnt++)).c_str());
                lua_pushstring(L, face["parsing"].get<std::string>().c_str());
            }

            lua_settable(L, -3);

            lua_pushstring(L, "landmark");
            lua_createtable(L, 0, face["landmark"].size());
            int _cnt = 0;
            for(auto ldm: face["landmark"]){
                _cnt++;
                lua_pushnumber(L, ldm[0]);
                lua_setfield(L, -2, ("x"+std::to_string(_cnt)).c_str());
                lua_pushnumber(L, ldm[1]);
                lua_setfield(L, -2, ("y"+std::to_string(_cnt)).c_str());
            }
            lua_settable(L, -3);

            if(face.contains("pose")){
                lua_pushstring(L, "pose");
                lua_createtable(L, 0, 3);
                lua_pushnumber(L, face["pose"][0]);
                lua_setfield(L, -2, "yaw");
                lua_pushnumber(L, face["pose"][1]);
                lua_setfield(L, -2, "pitch");
                lua_pushnumber(L, face["pose"][2]);
                lua_setfield(L, -2, "roll");

                lua_settable(L, -3);
            }

            lua_settable(L, -3);
        }

        lua_settable(L, -3);
    }

    if(output.contains("object")) {
        lua_pushstring(L, "object");
        lua_createtable(L, output["object"].size(), 0);
        int cnt = 0;
        for(auto object: output["object"]){
            lua_pushstring(L, ("object" + std::to_string(cnt++)).c_str());
            lua_newtable(L);

            lua_pushnumber(L, object["conf"]);
            lua_setfield(L, -2, "conf");

            lua_pushnumber(L, object["label"]);
            lua_setfield(L, -2, "label");

            lua_pushstring(L, "bbox");
            lua_createtable(L, 0, 4);

            lua_pushnumber(L, object["bbox"][0]);
            lua_setfield(L, -2, "x1");

            lua_pushnumber(L, object["bbox"][1]);
            lua_setfield(L, -2, "y1");

            lua_pushnumber(L, object["bbox"][2]);
            lua_setfield(L, -2, "x2");

            lua_pushnumber(L, object["bbox"][3]);
            lua_setfield(L, -2, "y2");

            lua_settable(L, -3);

            lua_settable(L, -3);
        }

        lua_settable(L, -3);
    }

    if(output.contains("ocr")) {
        lua_pushstring(L, "ocr");
        lua_createtable(L, output["ocr"].size(), 0);
        int cnt = 0;
        for(auto ocr: output["ocr"]){
            lua_pushstring(L, ("ocr" + std::to_string(cnt++)).c_str());
            lua_newtable(L);

            lua_pushnumber(L, ocr["conf"]);
            lua_setfield(L, -2, "conf");

            lua_pushnumber(L, ocr["conf_rotate"]);
            lua_setfield(L, -2, "conf_rotate");

            lua_pushstring(L, ocr["label"].get<std::string>().c_str());
            lua_setfield(L, -2, "label");

            lua_pushstring(L, "box");

            lua_createtable(L, 0, 8);

            int _cnt = 0;
            for(auto xy: ocr["box"]){
                _cnt++;
                lua_pushnumber(L, xy[0]);
                lua_setfield(L, -2, ("x"+std::to_string(_cnt)).c_str());
                lua_pushnumber(L, xy[1]);
                lua_setfield(L, -2, ("y"+std::to_string(_cnt)).c_str());
            }
            lua_settable(L, -3);

            lua_settable(L, -3);

        }

        lua_settable(L, -3);
    }

    if(output.contains("key_points")) {
        lua_pushstring(L, "key_points");
        lua_createtable(L, output["key_points"].size() * 2, 0);
        int _cnt = 0;
        for(auto kps: output["key_points"]){
            _cnt++;
            lua_pushnumber(L, kps[0]);
            lua_setfield(L, -2, ("x"+std::to_string(_cnt)).c_str());
            lua_pushnumber(L, kps[1]);
            lua_setfield(L, -2, ("y"+std::to_string(_cnt)).c_str());
        }

        lua_settable(L, -3);
    }

    if(output.contains("cls")) {
        lua_pushstring(L, "cls");
        lua_createtable(L, output["cls"].size(), 0);
        int _cnt = 0;
        for(auto cls: output["cls"]){

            lua_pushstring(L, ("top" + std::to_string(++_cnt)).c_str());
            lua_newtable(L);

            lua_pushnumber(L, cls["conf"]);
            lua_setfield(L, -2, "conf");
            lua_pushnumber(L, cls["label"]);
            lua_setfield(L, -2, "label");

            lua_settable(L, -3);
        }

        lua_settable(L, -3);
    }


//    std::cout << output << std::endl;

    delete []binary_result;

    return 1;
}


//使用luaL_Reg注册新的C函数到Lua中
static luaL_Reg aidb_functions[] =
        {
            {"AiDBCreate", AiDBCreate},
            {"AiDBFree", AiDBFree},
            {"AiDBRegister", AiDBRegister},
            {"AiDBUnRegister", AiDBUnRegister},
            {"AiDBForward", AiDBForward},
            {nullptr, nullptr}   //数组中最后一对必须是{NULL, NULL}，用来表示结束
    };


/* luaopen_XXX，XXX为库名称，若库名称为testlib.so，XXX即为testlib */
int luaopen_aidb(lua_State *L) {
    luaL_newlib(L, aidb_functions);  //Lua 5.2之后用luaL_newlib代替了luaL_register
    return 1;

}

//using namespace kaguya;
//
//int g_size = 0;
//typedef void* AiDB;
//AiDB LuaAiDBCreate();
//void LuaAiDBFree(AiDB ins);
//int LuaAiDBRegister(AiDB ins,const char* parameter);
//int LuaAiDBUnRegister(AiDB ins, const char* flow_uuid);
//int LuaAiDBForward(AiDB ins, const char* flow_uuid, const char* binary_image, int cache = 1024 * 1024);
//
//AiDB LuaAiDBCreate(){
//    AIDB::aidb_log_init(AIDB_DEBUG, "debug");
//    auto aidb_ins = new AiDBServer::AiDBServer();
//    std::cout << "LuaAiDBCreate\n";
//    return (void*)aidb_ins;
//}
//
//void LuaAiDBFree(AiDB ins){
//    if(nullptr != ins){
//        auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
//        delete aidb_ins;
//        aidb_ins = nullptr;
//    }
//}
//
//int LuaAiDBRegister(AiDB ins,const char* parameter){
//    if(nullptr == ins){
//        std::cout << "AiDBRegister null \n";
//        return -1;
//    }
//    auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
//    return aidb_ins->Register(parameter);
//}
//int LuaAiDBUnRegister(AiDB ins, const char* flow_uuid){
//    if(nullptr == ins){
//        std::cout << "AiDBUnRegister null \n";
//        return -1;
//    }
//    auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
//    return aidb_ins->unRegister(flow_uuid);
//}
//
//
//int LuaAiDBForward(AiDB ins, const char* flow_uuid, const char* binary_image, int cache){
//    if(nullptr == ins){
//        std::cout << "AiDBForward null \n";
//        return -1;
//    }
//
//    auto aidb_ins = static_cast<AiDBServer::AiDBServer*>(ins);
//    char* binary_result = new char[cache];
//    int size_out;
//
//    int ret =  aidb_ins->Forward(binary_image, flow_uuid, binary_result, cache, &size_out);
//
//    json output = json::parse(binary_result);
//
//    g_size = output.size();
//    std::cout << "@@@@@" << g_size << std::endl;
//    delete []binary_result;
//
//    return ret;
//}
//
//
//
//KAGUYA_FUNCTION_OVERLOADS(LuaAiDBForward_wrapper, LuaAiDBForward,3,4);
//
//
//
//
//int luaopen_aidb(lua_State *L){
//    kaguya::State state(L);
//    kaguya::LuaTable module = state.newTable();
//    module["AiDBCreate"] = &LuaAiDBCreate;
//    module["AiDBFree"] = &LuaAiDBFree;
//    module["AiDBRegister"] = &LuaAiDBRegister;
//    module["AiDBUnRegister"] = &LuaAiDBUnRegister;
//
//    module["AiDBForward"] = kaguya::function(LuaAiDBForward_wrapper());
//
//    module["size"] = g_size;
//
//    return module.push();
//
//}

//"\"flow_uuid\": \"flow_uuid\", \"backend\":\"mnn\", \"model\":\"scrfd_500m_kps\", \"zoo\":\"./config\""

// aidb = require("aidb")
// aidb.AiDBCreate