//
// Created by TalkUHulk on 2023/8/1.
//

#ifndef AIDB_WRAPPER_H
#define AIDB_WRAPPER_H

//#include "kaguya/kaguya.hpp"
//#define KAGUYA_DYNAMIC_LIB
//#include <lua.hpp>
extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

#ifdef __cplusplus
extern "C" {
#endif

//typedef void* AiDB;

//int AiDBCreate(lua_State *L);
//void AiDBFree(lua_State *L);
//int AiDBRegister(lua_State *L);
//int AiDBUnRegister(lua_State *L);
//int AiDBForward(lua_State *L);

int luaopen_aidb(lua_State *L);

#ifdef __cplusplus
}
#endif

#endif //AIDB_WRAPPER_H
