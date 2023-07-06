//
// Created by TalkUHulk on 2023/6/20.
//

#ifndef AIDB_AIDB_C_API_H
#define AIDB_AIDB_C_API_H

#ifdef __cplusplus
extern "C" {
#endif
    typedef void* AiDB;
    AiDB AiDBCreate();
    void AiDBFree(AiDB ins);
    int AiDBRegister(AiDB ins,const char* parameter);
    int AiDBUnRegister(AiDB ins, const char* flow_uuid);
    int AiDBForward(AiDB ins, const char* flow_uuid, const char* binary_image, char* binary_result, int size_in, int* size_out);
#ifdef __cplusplus
}
#endif

#endif //AIDB_AIDB_C_API_H
