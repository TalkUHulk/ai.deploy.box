//
// Created by TalkUHulk on 2023/5/4.
//

#include <jni.h>
#include <string>
#include <vector>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include "core/Interpreter.h"
#include "utility/Utility.h"
#include "utility/Logging.h"
#include "utility/td_obj.h"
#include "utility/face_align.h"
#include <android/bitmap.h>
#include <memory>
#include <chrono>
#include <thread>

AIDB::Interpreter *g_interpreter = nullptr;
AIDB::Interpreter *g_interpreter1 = nullptr;
AIDB::Interpreter *g_interpreter2 = nullptr;
std::string g_model_str;
//std::string g_work_home;
bool running = false;


extern "C"
JNIEXPORT void JNICALL
Java_com_hulk_aidb_1demo_ProcessActivity_aidbCreateInstance(JNIEnv *env, jobject thiz, jstring model_name,
                                                            jstring backend, jstring config)  {
    const char *_model_name = env->GetStringUTFChars(model_name, nullptr);
    const char *_backend = env->GetStringUTFChars(backend, nullptr);
    const char *_config = env->GetStringUTFChars(config, nullptr);

    g_model_str = std::string(_model_name);

    if(g_interpreter != nullptr){
        AIDB::Interpreter::releaseInstance(g_interpreter);
        g_interpreter = nullptr;
    }
    if(g_interpreter1 != nullptr){
        AIDB::Interpreter::releaseInstance(g_interpreter1);
        g_interpreter1 = nullptr;
    }
    if(g_interpreter2 != nullptr){
        AIDB::Interpreter::releaseInstance(g_interpreter2);
        g_interpreter2 = nullptr;
    }


    if(g_model_str == "pfpld" || g_model_str.find("3ddfa") != std::string::npos || g_model_str == "bisenet"){
        g_interpreter = AIDB::Interpreter::createInstance("scrfd_500m_kps", _backend, _config);
        g_interpreter1 = AIDB::Interpreter::createInstance(_model_name, _backend, _config);
    } else if(g_model_str == "ppocr"){
        g_interpreter = AIDB::Interpreter::createInstance("ppocr_det", _backend, _config);
        g_interpreter1 = AIDB::Interpreter::createInstance("ppocr_cls", _backend, _config);
        g_interpreter2 = AIDB::Interpreter::createInstance("ppocr_ret", _backend, _config);

    } else{
        g_interpreter = AIDB::Interpreter::createInstance(_model_name, _backend, _config);
        LOGD("@@@@@@@@@", "g_interpreter = %d", g_interpreter == nullptr);
    }

    size_t index = std::string(_config).rfind('/');
//    g_work_home = std::string(_config).substr(0, index);

}


extern "C"
JNIEXPORT void JNICALL
Java_com_hulk_aidb_1demo_ProcessActivity_aidbReleaseInstance(JNIEnv *env, jobject thiz) {

    LOGD("====>> destory1:", "%d", running);
    while(running){
        std::this_thread::sleep_for (std::chrono::milliseconds (1));
    }
    LOGD("====>> destory2:", "%d", running);
    AIDB::Interpreter::releaseInstance(g_interpreter);
    AIDB::Interpreter::releaseInstance(g_interpreter1);
    AIDB::Interpreter::releaseInstance(g_interpreter2);

    g_interpreter = nullptr;
    g_interpreter1 = nullptr;
    g_interpreter2 = nullptr;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_hulk_aidb_1demo_ProcessActivity_aidbForward(
        JNIEnv* env,
        jobject thiz,
        jobject bitmap) {

    assert(g_interpreter != nullptr );

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    AndroidBitmapInfo bitmapInfo;
    void * bitmapPixels;
    cv::Mat frame;
    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) >= 0);
    CV_Assert(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) >= 0);
    CV_Assert(bitmapPixels);
    LOGD("@@@@@@ forward", "width%d, height:%d", bitmapInfo.width, bitmapInfo.height);

    running = true;
    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);    // Establish temporary mat
        tmp.copyTo(frame);                                                         // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, frame, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

    //C++结构体对象转换为java对象；
    // 1）获取java ReturnInfo对象的jclass；
    jclass jAIDBMetaClass = env->FindClass("com/hulk/aidb/AIDBMeta");
    jobject jAIDBMetaObj = nullptr;

    if(nullptr == g_interpreter){
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(I)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, -1);
        return jAIDBMetaObj;
    }
    if(g_model_str.find("scrfd") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 0);

        jfieldID jFaceMetaID      = (env)->GetFieldID(jAIDBMetaClass, "face_meta", "[Lcom/hulk/aidb/FaceMeta;");

        jclass jFaceMetaClass = env->FindClass("com/hulk/aidb/FaceMeta");
        jfieldID jLandMarks = (env)->GetFieldID(jFaceMetaClass, "landmarks", "[F");
        jobjectArray jFaceMetaArray = env->NewObjectArray((int)face_metas.size(), jFaceMetaClass, nullptr);
        for(int i = 0; i < face_metas.size(); i++){
            jmethodID jFaceMetaConstructor = env->GetMethodID(jFaceMetaClass, "<init>", "(FFFFF)V");
            jobject jFaceMetaObj = env->NewObject(jFaceMetaClass, jFaceMetaConstructor,
                                                  face_metas[i]->x1, face_metas[i]->y1,
                                                  face_metas[i]->x2, face_metas[i]->y2, face_metas[i]->score);

            jfloatArray LDMArray = env->NewFloatArray(10);
            env->SetFloatArrayRegion(LDMArray, 0, 10, (jfloat *)face_metas[i]->kps.data());
            env->SetObjectField(jFaceMetaObj, jLandMarks, LDMArray);

            env->SetObjectArrayElement(jFaceMetaArray, i, jFaceMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jFaceMetaID, jFaceMetaArray);
    }
    else if(g_model_str.find("pfpld") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 0);

        jfieldID jFaceMetaID      = (env)->GetFieldID(jAIDBMetaClass, "face_meta", "[Lcom/hulk/aidb/FaceMeta;");

        jclass jFaceMetaClass = env->FindClass("com/hulk/aidb/FaceMeta");
        jfieldID jLandMarks = (env)->GetFieldID(jFaceMetaClass, "landmarks", "[F");
        jobjectArray jFaceMetaArray = env->NewObjectArray((int)face_metas.size(), jFaceMetaClass, nullptr);
        for(int i = 0; i < face_metas.size(); i++){

            outputs.clear();
            outputs_shape.clear();
            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_metas[i], face_meta_roi, frame.cols, frame.rows, 1.28, 0.14);
            cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));
            blob = *g_interpreter1 << roi;
            g_interpreter1->forward((float*)blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            AIDB::Utility::pfpld_post_process(outputs, face_meta_roi, face_metas[i], 98);


            jmethodID jFaceMetaConstructor = env->GetMethodID(jFaceMetaClass, "<init>", "(FFFFFI)V");
            jobject jFaceMetaObj = env->NewObject(jFaceMetaClass, jFaceMetaConstructor,
                                                  face_metas[i]->x1, face_metas[i]->y1,
                                                  face_metas[i]->x2, face_metas[i]->y2,
                                                  face_metas[i]->score, 98);

            jfloatArray LDMArray = env->NewFloatArray(98 * 2);
            env->SetFloatArrayRegion(LDMArray, 0, 98 * 2, (jfloat *)face_metas[i]->kps.data());
            env->SetObjectField(jFaceMetaObj, jLandMarks, LDMArray);

            env->SetObjectArrayElement(jFaceMetaArray, i, jFaceMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jFaceMetaID, jFaceMetaArray);
    }
    else if(g_model_str.find("3ddfa") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 3);
        // only process max area face
        if(!face_metas.empty()){
            std::sort(face_metas.begin(), face_metas.end(),
                      [](std::shared_ptr<AIDB::FaceMeta> meta1, std::shared_ptr<AIDB::FaceMeta> meta2){
                return meta1->area() > meta2->area();
            });
            jmethodID jsetBitmapFlag  = env->GetMethodID(jAIDBMetaClass, "setBitmapFlag", "(ZZ)V");
            env->CallVoidMethod(jAIDBMetaObj, jsetBitmapFlag, true, true);
            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_metas[0], face_meta_roi, frame.cols, frame.rows, 1.58, 0.14);
            cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));

            blob = *g_interpreter1 << roi;
            outputs.clear();
            outputs_shape.clear();

            g_interpreter1->forward((float*)blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            std::vector<float> vertices, pose, sRt;

            AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

            start = std::chrono::system_clock::now();
            if(g_model_str.find("dense") != std::string::npos){
                AIDB::Utility::TddfaUtility::tddfa_rasterize(frame, vertices, spider_man_obj, 1, true);
            }
            else{
                for(int n = 0; n < vertices.size() / 3; n++){
                    cv::circle(frame, cv::Point(vertices[3*n], vertices[3*n+1]), 2, cv::Scalar(255, 255, 255), -1);
                }
                AIDB::Utility::TddfaUtility::plot_pose_box(frame, sRt, vertices, 68);
            }


            jclass bitmapConfig = env->FindClass("android/graphics/Bitmap$Config");
            jfieldID rgba8888FieldID = env->GetStaticFieldID(bitmapConfig, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
            jobject rgba8888Obj = env->GetStaticObjectField(bitmapConfig, rgba8888FieldID);

            jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
            jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass,"createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
            jobject bitmapObj = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, frame.cols, frame.rows, rgba8888Obj);

            jintArray pixels = env->NewIntArray(frame.cols * frame.rows);
            for (int i = 0; i < frame.cols * frame.rows; i++){
                unsigned char red = frame.data[i * 3 + 2];
                unsigned char green = frame.data[i * 3 + 1];
                unsigned char blue = frame.data[i * 3];
                unsigned char alpha = 255;
                int currentPixel = (alpha << 24) | (red << 16) | (green << 8) | (blue);
                env->SetIntArrayRegion(pixels, i, 1, &currentPixel);
            }

            jmethodID setPixelsMid = env->GetMethodID(bitmapClass, "setPixels", "([IIIIIII)V");
            env->CallVoidMethod(bitmapObj, setPixelsMid, pixels, 0, frame.cols, 0, 0, frame.cols, frame.rows);

            jfieldID jBitmapID = (env)->GetFieldID(jAIDBMetaClass, "bitmap","Landroid/graphics/Bitmap;");
            env->SetObjectField(jAIDBMetaObj, jBitmapID, bitmapObj);
        }
    }
    else if(g_model_str.find("yolox") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
        auto post_process = AIDB::Utility::YoloX(g_interpreter->width(), 0.25, 0.45, {8, 16, 32});
        post_process(outputs[0], outputs_shape[0], object, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        LOGD("@@@@@@ forward", "yolox object num:%lu", object.size());

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)object.size(), 1);

        jfieldID jObjectMetaID      = (env)->GetFieldID(jAIDBMetaClass, "object_meta", "[Lcom/hulk/aidb/ObjectMeta;");

        jclass jObjectMetaClass = env->FindClass("com/hulk/aidb/ObjectMeta");
        jobjectArray jObjectMetaArray = env->NewObjectArray((int)object.size(), jObjectMetaClass, nullptr);
        for(int i = 0; i < object.size(); i++){
            jmethodID jObjectConstructor = env->GetMethodID(jObjectMetaClass, "<init>", "(FFFFFI)V");
            jobject jObjectMetaObj = env->NewObject(jObjectMetaClass, jObjectConstructor,
                                                    object[i]->x1, object[i]->y1,
                                                    object[i]->x2, object[i]->y2,
                                                    object[i]->score, object[i]->label);

            env->SetObjectArrayElement(jObjectMetaArray, i, jObjectMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jObjectMetaID, jObjectMetaArray);

    }
    else if(g_model_str.find("yolov7") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

        if(g_model_str.find("grid") != std::string::npos){
            AIDB::Utility::yolov7_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.25, g_interpreter->scale_h());
        } else{
            AIDB::Utility::yolov7_post_process(outputs, outputs_shape, object, 0.45, 0.25, g_interpreter->scale_h());
        }

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)object.size(), 1);

        jfieldID jObjectMetaID      = (env)->GetFieldID(jAIDBMetaClass, "object_meta", "[Lcom/hulk/aidb/ObjectMeta;");

        jclass jObjectMetaClass = env->FindClass("com/hulk/aidb/ObjectMeta");
        jobjectArray jObjectMetaArray = env->NewObjectArray((int)object.size(), jObjectMetaClass, nullptr);
        for(int i = 0; i < object.size(); i++){
            jmethodID jObjectConstructor = env->GetMethodID(jObjectMetaClass, "<init>", "(FFFFFI)V");
            jobject jObjectMetaObj = env->NewObject(jObjectMetaClass, jObjectConstructor,
                                                    object[i]->x1, object[i]->y1,
                                                    object[i]->x2, object[i]->y2,
                                                    object[i]->score, object[i]->label);

            env->SetObjectArrayElement(jObjectMetaArray, i, jObjectMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jObjectMetaID, jObjectMetaArray);

    }
    else if(g_model_str.find("yolov8") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

        AIDB::Utility::yolov8_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.35, g_interpreter->scale_h());

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)object.size(), 1);

        jfieldID jObjectMetaID      = (env)->GetFieldID(jAIDBMetaClass, "object_meta", "[Lcom/hulk/aidb/ObjectMeta;");

        jclass jObjectMetaClass = env->FindClass("com/hulk/aidb/ObjectMeta");
        jobjectArray jObjectMetaArray = env->NewObjectArray((int)object.size(), jObjectMetaClass, nullptr);
        for(int i = 0; i < object.size(); i++){
            jmethodID jObjectConstructor = env->GetMethodID(jObjectMetaClass, "<init>", "(FFFFFI)V");
            jobject jObjectMetaObj = env->NewObject(jObjectMetaClass, jObjectConstructor,
                                                    object[i]->x1, object[i]->y1,
                                                    object[i]->x2, object[i]->y2,
                                                    object[i]->score, object[i]->label);

            env->SetObjectArrayElement(jObjectMetaArray, i, jObjectMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jObjectMetaID, jObjectMetaArray);

    }
    else if(g_model_str.find("bisenet") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 3);
        // only process max area face
        if(!face_metas.empty()){
            std::sort(face_metas.begin(), face_metas.end(),
                      [](std::shared_ptr<AIDB::FaceMeta> meta1, std::shared_ptr<AIDB::FaceMeta> meta2){
                          return meta1->area() > meta2->area();
                      });
            jmethodID jsetBitmapFlag  = env->GetMethodID(jAIDBMetaClass, "setBitmapFlag", "(ZZ)V");
            env->CallVoidMethod(jAIDBMetaObj, jsetBitmapFlag, true, false);

            cv::Mat align;
            AIDB::faceAlign(frame, align, face_metas[0], g_interpreter1->width(), "ffhq");
            blob = *g_interpreter1 << align;
            outputs.clear();
            outputs_shape.clear();

            g_interpreter1->forward((float*)blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            start = std::chrono::system_clock::now();
            AIDB::Utility::bisenet_post_process(align, align, outputs[0], outputs_shape[0]);

            jclass bitmapConfig = env->FindClass("android/graphics/Bitmap$Config");
            jfieldID rgba8888FieldID = env->GetStaticFieldID(bitmapConfig, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
            jobject rgba8888Obj = env->GetStaticObjectField(bitmapConfig, rgba8888FieldID);

            jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
            jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass,"createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
            jobject bitmapObj = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, align.cols, align.rows, rgba8888Obj);

            jintArray pixels = env->NewIntArray(align.cols * align.rows);
            for (int i = 0; i < align.cols * align.rows; i++){
                unsigned char red = align.data[i * 3 + 2];
                unsigned char green = align.data[i * 3 + 1];
                unsigned char blue = align.data[i * 3];
                unsigned char alpha = 255;
                int currentPixel = (alpha << 24) | (red << 16) | (green << 8) | (blue);
                env->SetIntArrayRegion(pixels, i, 1, &currentPixel);
            }

            jmethodID setPixelsMid = env->GetMethodID(bitmapClass, "setPixels", "([IIIIIII)V");
            env->CallVoidMethod(bitmapObj, setPixelsMid, pixels, 0, align.cols, 0, 0, align.cols, align.rows);

            jfieldID jBitmapID = (env)->GetFieldID(jAIDBMetaClass, "bitmap","Landroid/graphics/Bitmap;");
            env->SetObjectField(jAIDBMetaObj, jBitmapID, bitmapObj);
        }
    }
    else if(g_model_str.find("movenet") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);

        cv::Mat result;
        std::vector<std::vector<float>> decoded_keypoints;
        AIDB::Utility::movenet_post_process(frame, outputs, outputs_shape, decoded_keypoints);
        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, 1, 0);

        jfieldID jFaceMetaID      = (env)->GetFieldID(jAIDBMetaClass, "face_meta", "[Lcom/hulk/aidb/FaceMeta;");

        jclass jFaceMetaClass = env->FindClass("com/hulk/aidb/FaceMeta");
        jfieldID jLandMarks = (env)->GetFieldID(jFaceMetaClass, "landmarks", "[F");
        jobjectArray jFaceMetaArray = env->NewObjectArray(1, jFaceMetaClass, nullptr);
        jmethodID jFaceMetaConstructor = env->GetMethodID(jFaceMetaClass, "<init>", "(I)V");
        jobject jFaceMetaObj = env->NewObject(jFaceMetaClass, jFaceMetaConstructor,
                                              (int)decoded_keypoints.size());
        jfloatArray LDMArray = env->NewFloatArray((int)decoded_keypoints.size() * 2);
        for(int i = 0; i < decoded_keypoints.size(); i++){
            env->SetFloatArrayRegion(LDMArray, i * 2, 2, (jfloat *)decoded_keypoints[i].data());
        }
        env->SetObjectField(jFaceMetaObj, jLandMarks, LDMArray);
        env->SetObjectArrayElement(jFaceMetaArray, 0, jFaceMetaObj);
        env->SetObjectField(jAIDBMetaObj, jFaceMetaID, jFaceMetaArray);
    }
    else if(g_model_str.find("ppocr") != std::string::npos){
//        cv::resize(frame, frame, cv::Size(240, 320));
        std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::OcrMeta>> ocr_results;

        auto post_process = AIDB::Utility::PPOCR(0.3, 0.5,
                                                 2.0, false, "fast", 0.9,
                                                 "/storage/emulated/0/Android/obb/com.hulk.aidb_demo/config/ppocr_keys_v1.txt");
        post_process.dbnet_post_process(outputs[0], outputs_shape[0], ocr_results, g_interpreter->scale_h(), g_interpreter->scale_w(), frame);

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)ocr_results.size(), 2);

        jfieldID jOcrMetaID      = (env)->GetFieldID(jAIDBMetaClass, "ocr_meta", "[Lcom/hulk/aidb/OcrMeta;");

        jclass jOcrMetaClass = env->FindClass("com/hulk/aidb/OcrMeta");
        jobjectArray jOcrMetaArray = env->NewObjectArray((int)ocr_results.size(), jOcrMetaClass, nullptr);

        for(int i = 0; i < ocr_results.size(); i++){

            cv::Mat crop_img;
            AIDB::Utility::PPOCR::GetRotateCropImage(frame, crop_img, ocr_results[i]);

            // Cls
            outputs.clear();
            outputs_shape.clear();
            cv::Mat cls_blob = *g_interpreter1 << crop_img;
            g_interpreter1->forward((float*)cls_blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            post_process.cls_post_process(outputs[0], outputs_shape[0], crop_img, crop_img, ocr_results[i]);

            // crnn
            outputs.clear();
            outputs_shape.clear();
            cv::Mat crnn_blob = *g_interpreter2 << crop_img;
            g_interpreter2->forward((float*)crnn_blob.data, g_interpreter2->width(), g_interpreter2->height(), g_interpreter2->channel(), outputs, outputs_shape);
            post_process.crnn_post_process(outputs[0], outputs_shape[0], ocr_results[i]);


            jmethodID jOcrConstructor = env->GetMethodID(jOcrMetaClass, "<init>", "(I)V");
            jobject jOcrMetaObj = env->NewObject(jOcrMetaClass, jOcrConstructor, 4);

            jfieldID jconf_rotate = env->GetFieldID(jOcrMetaClass, "conf_rotate", "F");
            env->SetFloatField(jOcrMetaObj, jconf_rotate, ocr_results[i]->conf_rotate);

            jfieldID jconf = env->GetFieldID(jOcrMetaClass, "conf", "F");
            env->SetFloatField(jOcrMetaObj, jconf, ocr_results[i]->conf);

            jfieldID jpoints = env->GetFieldID(jOcrMetaClass, "points", "[F");
            jfloatArray pointArray = env->NewFloatArray(ocr_results[i]->box.size() * 2);
            std::vector<float> points_vec;
            for(auto box: ocr_results[i]->box){
                points_vec.push_back(box._x);
                points_vec.push_back(box._y);
            }
            env->SetFloatArrayRegion(pointArray, 0, points_vec.size(), points_vec.data());
            env->SetObjectField(jOcrMetaObj, jpoints, pointArray);

            env->SetObjectArrayElement(jOcrMetaArray, i, jOcrMetaObj);

            jfieldID jtext = env->GetFieldID(jOcrMetaClass, "text", "Ljava/lang/String;");
            jstring ocr_str = env->NewStringUTF(ocr_results[i]->label.c_str());

            env->SetObjectField(jOcrMetaObj, jtext, ocr_str);

        }
        env->SetObjectField(jAIDBMetaObj, jOcrMetaID, jOcrMetaArray);
    }
    else if(g_model_str.find("mobilevit") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ClsMeta>> predicts;
        auto post_process = AIDB::Utility::ImageNet("/storage/emulated/0/Android/obb/com.hulk.aidb_demo/config/imagenet-1k-id.txt");
        post_process(outputs[0], outputs_shape[0], predicts, 1);

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, 0, 3);

        jfieldID jlabelID      = (env)->GetFieldID(jAIDBMetaClass, "label", "Ljava/lang/String;");
        jfieldID jconfID      = (env)->GetFieldID(jAIDBMetaClass, "conf", "F");
        jstring label_str = env->NewStringUTF(predicts[0]->label_str.c_str());

        env->SetObjectField(jAIDBMetaObj, jlabelID, label_str);
        env->SetFloatField(jAIDBMetaObj, jconfID, predicts[0]->conf);

    }

    running = false;
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    float time = elapsed.count() * 1000;
    jfieldID jCostTime = (env)->GetFieldID(jAIDBMetaClass, "time", "F");
    env->SetFloatField(jAIDBMetaObj, jCostTime, time);
    return jAIDBMetaObj;

}


extern "C"
JNIEXPORT jobject JNICALL
Java_com_hulk_aidb_1demo_CaptureActivity_aidbForward(
        JNIEnv* env,
        jobject thiz,
        jobject bitmap) {

    assert(g_interpreter != nullptr );

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    AndroidBitmapInfo bitmapInfo;
    void * bitmapPixels;
    cv::Mat frame;
    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) >= 0);
    CV_Assert(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels) >= 0);
    CV_Assert(bitmapPixels);
    LOGD("@@@@@@ forward", "width%d, height:%d", bitmapInfo.width, bitmapInfo.height);

    running = true;
    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);    // Establish temporary mat
        tmp.copyTo(frame);                                                         // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, frame, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

    //C++结构体对象转换为java对象；
    // 1）获取java ReturnInfo对象的jclass；
    jclass jAIDBMetaClass = env->FindClass("com/hulk/aidb/AIDBMeta");
    jobject jAIDBMetaObj = nullptr;

    if(nullptr == g_interpreter){
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(I)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, -1);
        return jAIDBMetaObj;
    }
    if(g_model_str.find("scrfd") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);

        start = std::chrono::system_clock::now();
        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        start = std::chrono::system_clock::now();
        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 0);

        jfieldID jFaceMetaID      = (env)->GetFieldID(jAIDBMetaClass, "face_meta", "[Lcom/hulk/aidb/FaceMeta;");

        jclass jFaceMetaClass = env->FindClass("com/hulk/aidb/FaceMeta");
        jfieldID jLandMarks = (env)->GetFieldID(jFaceMetaClass, "landmarks", "[F");
        jobjectArray jFaceMetaArray = env->NewObjectArray((int)face_metas.size(), jFaceMetaClass, nullptr);
        for(int i = 0; i < face_metas.size(); i++){
            jmethodID jFaceMetaConstructor = env->GetMethodID(jFaceMetaClass, "<init>", "(FFFFF)V");
            jobject jFaceMetaObj = env->NewObject(jFaceMetaClass, jFaceMetaConstructor,
                                                  face_metas[i]->x1, face_metas[i]->y1,
                                                  face_metas[i]->x2, face_metas[i]->y2, face_metas[i]->score);

            jfloatArray LDMArray = env->NewFloatArray(10);
            env->SetFloatArrayRegion(LDMArray, 0, 10, (jfloat *)face_metas[i]->kps.data());
            env->SetObjectField(jFaceMetaObj, jLandMarks, LDMArray);

            env->SetObjectArrayElement(jFaceMetaArray, i, jFaceMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jFaceMetaID, jFaceMetaArray);

    }
    else if(g_model_str.find("pfpld") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 0);

        jfieldID jFaceMetaID      = (env)->GetFieldID(jAIDBMetaClass, "face_meta", "[Lcom/hulk/aidb/FaceMeta;");

        jclass jFaceMetaClass = env->FindClass("com/hulk/aidb/FaceMeta");
        jfieldID jLandMarks = (env)->GetFieldID(jFaceMetaClass, "landmarks", "[F");
        jobjectArray jFaceMetaArray = env->NewObjectArray((int)face_metas.size(), jFaceMetaClass, nullptr);
        for(int i = 0; i < face_metas.size(); i++){

            outputs.clear();
            outputs_shape.clear();
            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_metas[i], face_meta_roi, frame.cols, frame.rows, 1.28, 0.14);
            cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));
            blob = *g_interpreter1 << roi;
            g_interpreter1->forward((float*)blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            AIDB::Utility::pfpld_post_process(outputs, face_meta_roi, face_metas[i], 98);


            jmethodID jFaceMetaConstructor = env->GetMethodID(jFaceMetaClass, "<init>", "(FFFFFI)V");
            jobject jFaceMetaObj = env->NewObject(jFaceMetaClass, jFaceMetaConstructor,
                                                  face_metas[i]->x1, face_metas[i]->y1,
                                                  face_metas[i]->x2, face_metas[i]->y2,
                                                  face_metas[i]->score, 98);

            jfloatArray LDMArray = env->NewFloatArray(98 * 2);
            env->SetFloatArrayRegion(LDMArray, 0, 98 * 2, (jfloat *)face_metas[i]->kps.data());
            env->SetObjectField(jFaceMetaObj, jLandMarks, LDMArray);

            env->SetObjectArrayElement(jFaceMetaArray, i, jFaceMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jFaceMetaID, jFaceMetaArray);
    }
    else if(g_model_str.find("3ddfa") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());
        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 3);
        // only process max area face
        if(!face_metas.empty()){
            std::sort(face_metas.begin(), face_metas.end(),
                      [](std::shared_ptr<AIDB::FaceMeta> meta1, std::shared_ptr<AIDB::FaceMeta> meta2){
                          return meta1->area() > meta2->area();
                      });
            jmethodID jsetBitmapFlag  = env->GetMethodID(jAIDBMetaClass, "setBitmapFlag", "(ZZ)V");
            env->CallVoidMethod(jAIDBMetaObj, jsetBitmapFlag, true, true);
            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_metas[0], face_meta_roi, frame.cols, frame.rows, 1.58, 0.14);
            cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));

            blob = *g_interpreter1 << roi;
            outputs.clear();
            outputs_shape.clear();
            start = std::chrono::system_clock::now();

            g_interpreter1->forward((float*)blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            std::vector<float> vertices, pose, sRt;

            AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

            if(g_model_str.find("dense") != std::string::npos){
                AIDB::Utility::TddfaUtility::tddfa_rasterize(frame, vertices, spider_man_obj, 1, true);
            }
            else{
                for(int n = 0; n < vertices.size() / 3; n++){
                    cv::circle(frame, cv::Point(vertices[3*n], vertices[3*n+1]), 2, cv::Scalar(255, 255, 255), -1);
                }
                AIDB::Utility::TddfaUtility::plot_pose_box(frame, sRt, vertices, 68);
            }


            jclass bitmapConfig = env->FindClass("android/graphics/Bitmap$Config");
            jfieldID rgba8888FieldID = env->GetStaticFieldID(bitmapConfig, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
            jobject rgba8888Obj = env->GetStaticObjectField(bitmapConfig, rgba8888FieldID);

            jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
            jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass,"createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
            jobject bitmapObj = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, frame.cols, frame.rows, rgba8888Obj);

            jintArray pixels = env->NewIntArray(frame.cols * frame.rows);
            for (int i = 0; i < frame.cols * frame.rows; i++){
                unsigned char red = frame.data[i * 3 + 2];
                unsigned char green = frame.data[i * 3 + 1];
                unsigned char blue = frame.data[i * 3];
                unsigned char alpha = 255;
                int currentPixel = (alpha << 24) | (red << 16) | (green << 8) | (blue);
                env->SetIntArrayRegion(pixels, i, 1, &currentPixel);
            }

            jmethodID setPixelsMid = env->GetMethodID(bitmapClass, "setPixels", "([IIIIIII)V");
            env->CallVoidMethod(bitmapObj, setPixelsMid, pixels, 0, frame.cols, 0, 0, frame.cols, frame.rows);

            jfieldID jBitmapID = (env)->GetFieldID(jAIDBMetaClass, "bitmap","Landroid/graphics/Bitmap;");
            env->SetObjectField(jAIDBMetaObj, jBitmapID, bitmapObj);
        }
    }
    else if(g_model_str.find("yolox") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
        auto post_process = AIDB::Utility::YoloX(g_interpreter->width(), 0.25, 0.45, {8, 16, 32});
        post_process(outputs[0], outputs_shape[0], object, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)object.size(), 1);

        jfieldID jObjectMetaID      = (env)->GetFieldID(jAIDBMetaClass, "object_meta", "[Lcom/hulk/aidb/ObjectMeta;");

        jclass jObjectMetaClass = env->FindClass("com/hulk/aidb/ObjectMeta");
        jobjectArray jObjectMetaArray = env->NewObjectArray((int)object.size(), jObjectMetaClass, nullptr);
        for(int i = 0; i < object.size(); i++){
            jmethodID jObjectConstructor = env->GetMethodID(jObjectMetaClass, "<init>", "(FFFFFI)V");
            jobject jObjectMetaObj = env->NewObject(jObjectMetaClass, jObjectConstructor,
                                                    object[i]->x1, object[i]->y1,
                                                    object[i]->x2, object[i]->y2,
                                                    object[i]->score, object[i]->label);

            env->SetObjectArrayElement(jObjectMetaArray, i, jObjectMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jObjectMetaID, jObjectMetaArray);

    }
    else if(g_model_str.find("yolov7") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

        if(g_model_str.find("grid") != std::string::npos){
            AIDB::Utility::yolov7_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.25, g_interpreter->scale_h());
        } else{
            AIDB::Utility::yolov7_post_process(outputs, outputs_shape, object, 0.45, 0.25, g_interpreter->scale_h());
        }

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)object.size(), 1);

        jfieldID jObjectMetaID      = (env)->GetFieldID(jAIDBMetaClass, "object_meta", "[Lcom/hulk/aidb/ObjectMeta;");

        jclass jObjectMetaClass = env->FindClass("com/hulk/aidb/ObjectMeta");
        jobjectArray jObjectMetaArray = env->NewObjectArray((int)object.size(), jObjectMetaClass, nullptr);
        for(int i = 0; i < object.size(); i++){
            jmethodID jObjectConstructor = env->GetMethodID(jObjectMetaClass, "<init>", "(FFFFFI)V");
            jobject jObjectMetaObj = env->NewObject(jObjectMetaClass, jObjectConstructor,
                                                    object[i]->x1, object[i]->y1,
                                                    object[i]->x2, object[i]->y2,
                                                    object[i]->score, object[i]->label);

            env->SetObjectArrayElement(jObjectMetaArray, i, jObjectMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jObjectMetaID, jObjectMetaArray);

    }
    else if(g_model_str.find("yolov8") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

        AIDB::Utility::yolov8_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.35, g_interpreter->scale_h());

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)object.size(), 1);

        jfieldID jObjectMetaID      = (env)->GetFieldID(jAIDBMetaClass, "object_meta", "[Lcom/hulk/aidb/ObjectMeta;");

        jclass jObjectMetaClass = env->FindClass("com/hulk/aidb/ObjectMeta");
        jobjectArray jObjectMetaArray = env->NewObjectArray((int)object.size(), jObjectMetaClass, nullptr);
        for(int i = 0; i < object.size(); i++){
            jmethodID jObjectConstructor = env->GetMethodID(jObjectMetaClass, "<init>", "(FFFFFI)V");
            jobject jObjectMetaObj = env->NewObject(jObjectMetaClass, jObjectConstructor,
                                                    object[i]->x1, object[i]->y1,
                                                    object[i]->x2, object[i]->y2,
                                                    object[i]->score, object[i]->label);

            env->SetObjectArrayElement(jObjectMetaArray, i, jObjectMetaObj);
        }
        env->SetObjectField(jAIDBMetaObj, jObjectMetaID, jObjectMetaArray);

    }
    else if(g_model_str.find("bisenet") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
        AIDB::Utility::scrfd_post_process(outputs, face_metas, g_interpreter->width(), g_interpreter->height(), g_interpreter->scale_h());

        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)face_metas.size(), 3);
        // only process max area face
        if(!face_metas.empty()){
            std::sort(face_metas.begin(), face_metas.end(),
                      [](std::shared_ptr<AIDB::FaceMeta> meta1, std::shared_ptr<AIDB::FaceMeta> meta2){
                          return meta1->area() > meta2->area();
                      });
            jmethodID jsetBitmapFlag  = env->GetMethodID(jAIDBMetaClass, "setBitmapFlag", "(ZZ)V");
            env->CallVoidMethod(jAIDBMetaObj, jsetBitmapFlag, true, false);

            cv::Mat align;
            AIDB::faceAlign(frame, align, face_metas[0], g_interpreter1->width(), "ffhq");
            blob = *g_interpreter1 << align;
            outputs.clear();
            outputs_shape.clear();
            start = std::chrono::system_clock::now();

            g_interpreter1->forward((float*)blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            start = std::chrono::system_clock::now();
            AIDB::Utility::bisenet_post_process(align, align, outputs[0], outputs_shape[0]);

            jclass bitmapConfig = env->FindClass("android/graphics/Bitmap$Config");
            jfieldID rgba8888FieldID = env->GetStaticFieldID(bitmapConfig, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
            jobject rgba8888Obj = env->GetStaticObjectField(bitmapConfig, rgba8888FieldID);

            jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
            jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass,"createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
            jobject bitmapObj = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, align.cols, align.rows, rgba8888Obj);

            jintArray pixels = env->NewIntArray(align.cols * align.rows);
            for (int i = 0; i < align.cols * align.rows; i++){
                unsigned char red = align.data[i * 3 + 2];
                unsigned char green = align.data[i * 3 + 1];
                unsigned char blue = align.data[i * 3];
                unsigned char alpha = 255;
                int currentPixel = (alpha << 24) | (red << 16) | (green << 8) | (blue);
                env->SetIntArrayRegion(pixels, i, 1, &currentPixel);
            }

            jmethodID setPixelsMid = env->GetMethodID(bitmapClass, "setPixels", "([IIIIIII)V");
            env->CallVoidMethod(bitmapObj, setPixelsMid, pixels, 0, align.cols, 0, 0, align.cols, align.rows);

            jfieldID jBitmapID = (env)->GetFieldID(jAIDBMetaClass, "bitmap","Landroid/graphics/Bitmap;");
            env->SetObjectField(jAIDBMetaObj, jBitmapID, bitmapObj);
        }
    }
    else if(g_model_str.find("movenet") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);

        cv::Mat result;
        std::vector<std::vector<float>> decoded_keypoints;
        AIDB::Utility::movenet_post_process(frame, outputs, outputs_shape, decoded_keypoints);
        // 2）获取构造方法ID；
        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        // 3）通过构造方法ID创建Java ReturnInfo对象；
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, 1, 0);

        jfieldID jFaceMetaID      = (env)->GetFieldID(jAIDBMetaClass, "face_meta", "[Lcom/hulk/aidb/FaceMeta;");

        jclass jFaceMetaClass = env->FindClass("com/hulk/aidb/FaceMeta");
        jfieldID jLandMarks = (env)->GetFieldID(jFaceMetaClass, "landmarks", "[F");
        jobjectArray jFaceMetaArray = env->NewObjectArray(1, jFaceMetaClass, nullptr);
        jmethodID jFaceMetaConstructor = env->GetMethodID(jFaceMetaClass, "<init>", "(I)V");
        jobject jFaceMetaObj = env->NewObject(jFaceMetaClass, jFaceMetaConstructor,
                                              (int)decoded_keypoints.size());
        jfloatArray LDMArray = env->NewFloatArray((int)decoded_keypoints.size() * 2);
        for(int i = 0; i < decoded_keypoints.size(); i++){
            env->SetFloatArrayRegion(LDMArray, i * 2, 2, (jfloat *)decoded_keypoints[i].data());
        }
        env->SetObjectField(jFaceMetaObj, jLandMarks, LDMArray);
        env->SetObjectArrayElement(jFaceMetaArray, 0, jFaceMetaObj);
        env->SetObjectField(jAIDBMetaObj, jFaceMetaID, jFaceMetaArray);
    }
    else if(g_model_str.find("ppocr") != std::string::npos){
//        cv::resize(frame, frame, cv::Size(240, 320));
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::OcrMeta>> ocr_results;

        auto post_process = AIDB::Utility::PPOCR(0.3, 0.5,
                                                 2.0, false, "fast", 0.9,
                                                 "/storage/emulated/0/Android/obb/com.hulk.aidb_demo/config/ppocr_keys_v1.txt");
        post_process.dbnet_post_process(outputs[0], outputs_shape[0], ocr_results, g_interpreter->scale_h(), g_interpreter->scale_w(), frame);

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, (int)ocr_results.size(), 2);

        jfieldID jOcrMetaID      = (env)->GetFieldID(jAIDBMetaClass, "ocr_meta", "[Lcom/hulk/aidb/OcrMeta;");

        jclass jOcrMetaClass = env->FindClass("com/hulk/aidb/OcrMeta");
        jobjectArray jOcrMetaArray = env->NewObjectArray((int)ocr_results.size(), jOcrMetaClass, nullptr);

        for(int i = 0; i < ocr_results.size(); i++){

            cv::Mat crop_img;
            AIDB::Utility::PPOCR::GetRotateCropImage(frame, crop_img, ocr_results[i]);

            // Cls
            outputs.clear();
            outputs_shape.clear();
            cv::Mat cls_blob = *g_interpreter1 << crop_img;
            g_interpreter1->forward((float*)cls_blob.data, g_interpreter1->width(), g_interpreter1->height(), g_interpreter1->channel(), outputs, outputs_shape);

            post_process.cls_post_process(outputs[0], outputs_shape[0], crop_img, crop_img, ocr_results[i]);

            // crnn
            outputs.clear();
            outputs_shape.clear();
            cv::Mat crnn_blob = *g_interpreter2 << crop_img;
            g_interpreter2->forward((float*)crnn_blob.data, g_interpreter2->width(), g_interpreter2->height(), g_interpreter2->channel(), outputs, outputs_shape);
            post_process.crnn_post_process(outputs[0], outputs_shape[0], ocr_results[i]);


            jmethodID jOcrConstructor = env->GetMethodID(jOcrMetaClass, "<init>", "(I)V");
            jobject jOcrMetaObj = env->NewObject(jOcrMetaClass, jOcrConstructor, 4);

            jfieldID jconf_rotate = env->GetFieldID(jOcrMetaClass, "conf_rotate", "F");
            env->SetFloatField(jOcrMetaObj, jconf_rotate, ocr_results[i]->conf_rotate);

            jfieldID jconf = env->GetFieldID(jOcrMetaClass, "conf", "F");
            env->SetFloatField(jOcrMetaObj, jconf, ocr_results[i]->conf);

            jfieldID jpoints = env->GetFieldID(jOcrMetaClass, "points", "[F");
            jfloatArray pointArray = env->NewFloatArray(ocr_results[i]->box.size() * 2);
            std::vector<float> points_vec;
            for(auto box: ocr_results[i]->box){
                points_vec.push_back(box._x);
                points_vec.push_back(box._y);
            }
            env->SetFloatArrayRegion(pointArray, 0, points_vec.size(), points_vec.data());
            env->SetObjectField(jOcrMetaObj, jpoints, pointArray);

            env->SetObjectArrayElement(jOcrMetaArray, i, jOcrMetaObj);

            jfieldID jtext = env->GetFieldID(jOcrMetaClass, "text", "Ljava/lang/String;");
            jstring ocr_str = env->NewStringUTF(ocr_results[i]->label.c_str());

            env->SetObjectField(jOcrMetaObj, jtext, ocr_str);

        }
        env->SetObjectField(jAIDBMetaObj, jOcrMetaID, jOcrMetaArray);

    }
    else if(g_model_str.find("mobilevit") != std::string::npos){
        cv::Mat blob = *g_interpreter << frame;
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;
        g_interpreter->forward((float*)blob.data, g_interpreter->width(), g_interpreter->height(), g_interpreter->channel(), outputs, outputs_shape);
        std::vector<std::shared_ptr<AIDB::ClsMeta>> predicts;
        auto post_process = AIDB::Utility::ImageNet("/storage/emulated/0/Android/obb/com.hulk.aidb_demo/config/imagenet-1k-id.txt");
        post_process(outputs[0], outputs_shape[0], predicts, 1);

        jmethodID jConstructor = env->GetMethodID(jAIDBMetaClass, "<init>", "(II)V");
        jAIDBMetaObj = env->NewObject(jAIDBMetaClass, jConstructor, 0, 3);

        jfieldID jlabelID      = (env)->GetFieldID(jAIDBMetaClass, "label", "Ljava/lang/String;");
        jfieldID jconfID      = (env)->GetFieldID(jAIDBMetaClass, "conf", "F");
        jstring label_str = env->NewStringUTF(predicts[0]->label_str.c_str());

        env->SetObjectField(jAIDBMetaObj, jlabelID, label_str);
        env->SetFloatField(jAIDBMetaObj, jconfID, predicts[0]->conf);

    }

    running = false;

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    float time = elapsed.count() * 1000;
    jfieldID jCostTime = (env)->GetFieldID(jAIDBMetaClass, "time", "F");
    env->SetFloatField(jAIDBMetaObj, jCostTime, time);
    return jAIDBMetaObj;

}