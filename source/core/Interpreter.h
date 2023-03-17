//
// Created by TalkUHulk on 2022/10/19.
//

#ifndef AIENGINE_AIENGINEINTERPRETER_H
#define AIENGINE_AIENGINEINTERPRETER_H

#include "AIDBDefine.h"
#include <string>
//#include <yaml-cpp/yaml.h>
#include <memory>
#include <opencv2/opencv.hpp>

namespace AIDB {

    class Engine;
    class AIDBInput;

    class AIDB_PUBLIC Interpreter {

    public:
//        static Interpreter* createFromNode(const YAML::Node& engine_mode);
        static Interpreter* createInstance(const std::string& model, const std::string& backend, const std::string& config_zoo="./config");
        static void releaseInstance(Interpreter* ins);
        void forward(const float *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape);
        ~Interpreter();

        cv::Mat operator << (const cv::Mat &image);
        cv::Mat operator << (const std::string &image_path);
        void operator >> (cv::Mat &dst);
        int width() const;
        int height() const;
        int channel() const;
        float scale_w() const;
        float scale_h() const;

    private:
        explicit Interpreter(Engine* engine);
        explicit Interpreter(Engine* engine, AIDBInput* input);
        AIDBInput* _ptr_input;
        Engine* _ptr_engine;


    };
}
#endif //AIENGINE_AIENGINEINTERPRETER_H
