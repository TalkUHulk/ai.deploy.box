//
// Created by TalkUHulk on 2022/10/19.
//

#ifndef AIENGINE_AIENGINEINTERPRETER_H
#define AIENGINE_AIENGINEINTERPRETER_H

#include "AIDBDefine.h"
#include <string>
#include <memory>
#ifdef ENABLE_NCNN_WASM
#include <simpleocv.h>
#else
#include <opencv2/opencv.hpp>
#endif

namespace AIDB {

    class Engine;
    class AIDBInput;

    class AIDB_PUBLIC Interpreter {

    public:
//        static Interpreter* createFromNode(const YAML::Node& engine_mode);
        static Interpreter* createInstance(const std::string& model, const std::string& backend, const std::string& config_zoo="./config");
#ifdef ENABLE_NCNN_WASM
        static Interpreter* createInstance(const void* buffer_in1, const void* buffer_in2, const std::string &config);
#endif
        static void releaseInstance(Interpreter* ins);
        ~Interpreter();
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape);

#ifndef ENABLE_NCNN_WASM
        cv::Mat operator << (const cv::Mat &image);
        cv::Mat operator << (const std::string &image_path);
#else
//        void forward(ncnn::Mat frame, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape);

        ncnn::Mat operator << (const cv::Mat &image);
        ncnn::Mat operator << (const std::string &image_path);
#endif
        void operator >> (cv::Mat &dst);
        int width() const;
        int height() const;
        int channel() const;
        float scale_w() const;
        float scale_h() const;
        void set_roi(cv::Rect2f roi);

        std::string which_model();
        std::string which_backend();

    private:
        explicit Interpreter(Engine* engine);
        explicit Interpreter(Engine* engine, AIDBInput* input);
        AIDBInput* _ptr_input;
        Engine* _ptr_engine;


    };
}
#endif //AIENGINE_AIENGINEINTERPRETER_H
