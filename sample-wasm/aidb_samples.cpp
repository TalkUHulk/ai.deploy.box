//
// Created by TalkUHulk on 2023/3/23.
//

#include "scrfd.h"
#include "pfpld.h"

static SCRFD* g_scrfd = 0;
static PFPLD* g_pfpld = 0;

static void on_image_render(cv::Mat& rgba)
{
    if (!g_scrfd)
    {
        g_scrfd = new SCRFD;

        g_scrfd->init();
    }

    if (!g_pfpld)
    {
        g_pfpld = new PFPLD;

        g_pfpld->init();
    }

    std::vector<std::shared_ptr<AIDB::FaceMeta>> result;
    g_scrfd->detect(rgba, result);
    g_pfpld->detect(rgba, result);
//    g_scrfd->draw(rgba, result);
    g_pfpld->draw(rgba, result);

}

//#ifdef __EMSCRIPTEN_PTHREADS__
//
//static const unsigned char* rgba_data = 0;
//static int w = 0;
//static int h = 0;
//
//static ncnn::Mutex lock;
//static ncnn::ConditionVariable condition;
//
//static ncnn::Mutex finish_lock;
//static ncnn::ConditionVariable finish_condition;
//
//static void worker(){
//    while (1){
//
//        lock.lock();
//        while (rgba_data == 0){
//            condition.wait(lock);
//        }
//
//        cv::Mat rgba(h, w, CV_8UC4, (void*)rgba_data);
//
//        on_image_render(rgba);
//
//        rgba_data = 0;
//
//        lock.unlock();
//
//        finish_lock.lock();
//        finish_condition.signal();
//        finish_lock.unlock();
//    }
//}
//
//#include <thread>
//static std::thread t(worker);
//
//extern "C" {
//
//void scrfd_ncnn(unsigned char* _rgba_data, int _w, int _h){
//    lock.lock();
//    while (rgba_data != 0)
//    {
//        condition.wait(lock);
//    }
//
//    rgba_data = _rgba_data;
//    w = _w;
//    h = _h;
//
//    lock.unlock();
//
//    condition.signal();
//
//    // wait for finished
//    finish_lock.lock();
//    while (rgba_data != 0){
//        finish_condition.wait(finish_lock);
//    }
//    finish_lock.unlock();
//}
//
//}
//
//#else // __EMSCRIPTEN_PTHREADS__
extern "C" {

void scrfd_ncnn(unsigned char* rgba_data, int w, int h)
{
    std::cout << "!!!!! scrfd_ncnn\n";
    cv::Mat rgba(h, w, CV_8UC4, (void*)rgba_data);

    on_image_render(rgba);
}

}

//#endif // __EMSCRIPTEN_PTHREADS__



