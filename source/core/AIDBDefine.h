//
// Created by TalkUHulk on 2022/10/18.
//

#ifndef AIENGINE_ENGINEDEFINE_H
#define AIENGINE_ENGINEDEFINE_H

namespace AIDB {

#define AI_PI   3.1415926535897932384626433832795

    enum EngineID {
        IDLE,
        ONNX = 1,
        MNN = 2,
        NCNN = 3,
        TNN = 4,
        OPENVINO = 5,
        PADDLE_LITE = 6,
        TRT = 7,
        TRITON = 8

    };  /*!< Engine类型*/

    enum Device { CPU = 0, GPU = 1}; /*!< 模型加载device*/

    typedef void(*register_preprocess_fn)(unsigned char *frame, int frame_width, int frame_height, ...);
    typedef void(*register_postprocess_fn)(const void *frame, int len, ...);

} //AIENGINEine

#if defined(_MSC_VER)
#if defined(BUILDING_AIENGINE_DLL)
#define AIDB_PUBLIC __declspec(dllexport)
#elif defined(USING_AIENGINE_DLL)
#define AIDB_PUBLIC __declspec(dllimport)
#else
#define AIDB_PUBLIC
#endif
#else
#define AIDB_PUBLIC __attribute__((visibility("default")))
#endif

#ifdef _USE_LOGCAT
#include <android/log.h>
#define ENGINE_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "ENGINEJNI", format, ##__VA_ARGS__)
#define ENGINE_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "ENGINEJNI", format, ##__VA_ARGS__)
#else
#define ENGINE_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define ENGINE_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define ENGINE_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            ENGINE_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define ENGINE_ASSERT(x)
#endif

#endif //AIENGINE_ENGINEDEFINE_H
