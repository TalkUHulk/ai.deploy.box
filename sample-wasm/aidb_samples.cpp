//
// Created by TalkUHulk on 2023/3/23.
//

#include "scrfd.h"
#include "pfpld.h"
#include "yolox.h"
#include "yolov7.h"
#include "yolov8.h"
#include "tddfa.h"
#include "movenet.h"

static SCRFD* g_scrfd = nullptr;
static PFPLD* g_pfpld = nullptr;
static YoloX* g_yolox = nullptr;
static YoloV7* g_yolov7 = nullptr;
static YoloV8* g_yolov8 = nullptr;
static TDDFA* g_tddfa = nullptr;
static MoveNet* g_movenet = nullptr;

static int g_model_type = -1;

static void on_image_render(cv::Mat& rgba, int model_type){
    if(g_model_type != model_type){
        switch (g_model_type){
            case 0:
                delete g_scrfd;
                g_scrfd = nullptr;
                break;
            case 1:
                delete g_scrfd;
                g_scrfd = nullptr;
                delete g_pfpld;
                g_pfpld = nullptr;
                break;
            case 2:
                delete g_scrfd;
                g_scrfd = nullptr;
                delete g_tddfa;
                g_tddfa = nullptr;
                break;
            case 3:
                delete g_yolov7;
                g_yolov7 = nullptr;
                break;
            case 4:
                delete g_yolov8;
                g_yolov8 = nullptr;
                break;
            case 5:
                delete g_yolox;
                g_yolox = nullptr;
                break;
            case 6:
                delete g_movenet;
                g_movenet = nullptr;
                break;
            default :
                std::cout << "first do\n";

        }
        g_model_type = model_type;
    }

    switch (g_model_type){
        case 0:{
            if (!g_scrfd){
                g_scrfd = new SCRFD;
                g_scrfd->init();
            }
            std::vector<std::shared_ptr<AIDB::FaceMeta>> result;
            g_scrfd->detect(rgba, result);
            g_scrfd->draw(rgba, result);
            }
            break;
        case 1: {
            if (!g_scrfd) {
                g_scrfd = new SCRFD;
                g_scrfd->init();
            }
            if (!g_pfpld) {
                g_pfpld = new PFPLD;
                g_pfpld->init();
            }
            std::vector<std::shared_ptr<AIDB::FaceMeta>> result;
            g_scrfd->detect(rgba, result);
            g_pfpld->detect(rgba, result);
            g_pfpld->draw(rgba, result);
            }
            break;
        case 2: {
            if (!g_scrfd) {
                g_scrfd = new SCRFD;
                g_scrfd->init();
            }
            if (!g_tddfa) {
                g_tddfa = new TDDFA;
                g_tddfa->init(true);
            }
            std::vector<std::shared_ptr<AIDB::FaceMeta>> result;
            g_scrfd->detect(rgba, result);
            std::vector<float> vertices, pose, sRt;
            g_tddfa->detect(rgba, result, vertices, pose, sRt);
            g_tddfa->draw(rgba, result, vertices, pose, sRt);
            }
            break;
        case 3: {
            if (!g_yolov7) {
                g_yolov7 = new YoloV7;
                g_yolov7->init();
            }
            std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
            g_yolov7->detect(rgba, object);
            g_yolov7->draw(rgba, object);
            }
            break;
        case 4: {
            if (!g_yolov8) {
                g_yolov8 = new YoloV8;
                g_yolov8->init();
            }
            std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
            g_yolov8->detect(rgba, object);
            g_yolov8->draw(rgba, object);
            }
            break;
        case 5: {
            if (!g_yolox) {
                g_yolox = new YoloX;
                g_yolox->init();
            }
            std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
            g_yolox->detect(rgba, object);
            g_yolox->draw(rgba, object);
            }
            break;
        case 6: {
            if (!g_movenet) {
                g_movenet = new MoveNet;
                g_movenet->init();
            }
            g_movenet->detect(rgba, rgba);
            }
            break;
        default :
            std::cout << "change model\n";

    }



}


//static void on_image_render(cv::Mat& rgba, )
//{
//    if (!g_scrfd)
//    {
//        g_scrfd = new SCRFD;
//
//        g_scrfd->init();
//    }
//
//    if (!g_pfpld)
//    {
//        g_pfpld = new PFPLD;
//
//        g_pfpld->init();
//    }
//
//    std::vector<std::shared_ptr<AIDB::FaceMeta>> result;
//    g_scrfd->detect(rgba, result);
//    g_pfpld->detect(rgba, result);
////    g_scrfd->draw(rgba, result);
//    g_pfpld->draw(rgba, result);

//    if (!g_yolox)
//    {
//        g_yolox = new YoloX;
//
//        g_yolox->init();
//    }
//    std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
//    g_yolox->detect(rgba, object);
//    g_yolox->draw(rgba, object);
//
//    if (!g_yolov7)
//    {
//        g_yolov7 = new YoloV7;
//
//        g_yolov7->init();
//    }
//    std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
//    g_yolov7->detect(rgba, object);
//    g_yolov7->draw(rgba, object);
//
//
//    if (!g_yolov8)
//    {
//        g_yolov8 = new YoloV8;
//
//        g_yolov8->init();
//    }
//    std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;
//    g_yolov8->detect(rgba, object);
//    g_yolov8->draw(rgba, object);
//
//    if (!g_tddfa)
//    {
//        g_tddfa = new TDDFA;
//
//        g_tddfa->init(true);
//    }
//
//    std::vector<float> vertices, pose, sRt;
//    g_tddfa->detect(rgba, result, vertices, pose, sRt);
//    g_tddfa->draw(rgba, result, vertices, pose, sRt);
//    if (!g_movenet)
//    {
//        g_movenet = new MoveNet;
//
//        g_movenet->init();
//    }
//    g_movenet->detect(rgba, rgba);
//}

#ifdef __EMSCRIPTEN_PTHREADS__

static const unsigned char* rgba_data = 0;
static int cur_model_type = -1;
static int w = 0;
static int h = 0;

static ncnn::Mutex lock;
static ncnn::ConditionVariable condition;

static ncnn::Mutex finish_lock;
static ncnn::ConditionVariable finish_condition;

static void worker(){
    while (1){

        lock.lock();
        while (rgba_data == 0){
            condition.wait(lock);
        }

        cv::Mat rgba(h, w, CV_8UC4, (void*)rgba_data);

        on_image_render(rgba, cur_model_type);

        rgba_data = 0;
        cur_model_type = -1;

        lock.unlock();

        finish_lock.lock();
        finish_condition.signal();
        finish_lock.unlock();
    }
}

#include <thread>
static std::thread t(worker);

extern "C" {

void aidb_wasm(unsigned char* _rgba_data, int _w, int _h, int _model_type){
//    std::cout << "aidb_wasm pthread\n";
    lock.lock();
    while (rgba_data != 0)
    {
        condition.wait(lock);
    }

    rgba_data = _rgba_data;
    w = _w;
    h = _h;
    cur_model_type = _model_type;

    lock.unlock();

    condition.signal();

    // wait for finished
    finish_lock.lock();
    while (rgba_data != 0){
        finish_condition.wait(finish_lock);
    }
    finish_lock.unlock();
}

}

#else // __EMSCRIPTEN_PTHREADS__
extern "C" {

void aidb_wasm(unsigned char* rgba_data, int w, int h, int model_type)
{
//    std::cout << "aidb_wasm\n";
    cv::Mat rgba(h, w, CV_8UC4, (void*)rgba_data);
    on_image_render(rgba, model_type);
}

}

#endif // __EMSCRIPTEN_PTHREADS__