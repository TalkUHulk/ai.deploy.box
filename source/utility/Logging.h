//
// Created by TalkUHulk on 2023/1/10.
//

#pragma once

#ifdef __ANDROID__
#include <android/log.h>
#else
#include <iostream>
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SHOWLOG
#ifdef __ANDROID__
#define LOGI(TAG,...)    __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__);
#define LOGW(TAG,...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__);
#define LOGE(TAG,...)  __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__);
#define LOGD(TAG,...)  __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__);

#else
#define LOGI(TAG,...) \
     do { \
             std::cout<<__FILE__<<" "<<__LINE__<<"line"<<" "<<TAG<<":  "; \
             printf(__VA_ARGS__); } \
     while (0)

#define LOGW(TAG,...) \
     do {  \
             std::cout<<__FILE__<<" "<<__LINE__<<"line"<<" "<<TAG<<":  "; \
             printf(__VA_ARGS__); } \
     while (0)

#define LOGE(TAG,...) \
     do {  \
       std::cout<<__FILE__<<" "<<__LINE__<<"line"<<" "<<TAG<<":  "; \
       printf(__VA_ARGS__); } \
     while (0)
#define LOGD(TAG,...) \
     do {  \
       std::cout<<__FILE__<<" "<<__LINE__<<"line"<<" "<<TAG<<":  "; \
       printf(__VA_ARGS__); } \
     while (0)
#endif

#else
#define LOGI(TAG,...)
#define LOGW(TAG,...)
#define LOGE(TAG,...)
#define LOGD(TAG,...)
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
