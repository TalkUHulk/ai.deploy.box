//
// Created by TalkUHulk on 2022/10/19.
//

#include "Interpreter.h"
#include "StatusCode.h"
#include "Engine.hpp"
#include "CoreConfig.hpp"
#include "Common.hpp"
#include "BaseInput.hpp"
#include <algorithm>
#include <preprocess/ImageInput.hpp>
#include "utility/log.h"
#include "utility/Utility.h"


namespace AIDB {



    Interpreter::~Interpreter() {
        if(nullptr != _ptr_engine)
            delete _ptr_engine;
        if(nullptr != _ptr_input)
            delete _ptr_input;
    }

    Interpreter::Interpreter(Engine *engine) {
        ENGINE_ASSERT(nullptr != engine)
        _ptr_engine = engine;
    }

    Interpreter::Interpreter(Engine *engine, AIDBInput *input) {
        ENGINE_ASSERT(nullptr != engine)
//        ENGINE_ASSERT(nullptr != input)
        _ptr_engine = engine;
        _ptr_input = input;
    }

    Interpreter *
    Interpreter::createInstance(const std::string& model, const std::string& backend, const std::string& config_zoo) {
        StatusCode status = NO_ERROR;
        aidb_log_init(AIDB_DEBUG, "debug");

        std::string backend_lower(backend);
        transform(backend.begin(),backend.end(),backend_lower.begin(),::tolower);

        std::string model_lower(model);
        transform(model.begin(),model.end(),model_lower.begin(),::tolower);
        transform(model.begin(),model.end(),model_lower.begin(),[=](char s){return '-' == s?'_':s;});

        std::string config_prefix(config_zoo);
        if(config_prefix.back() != '/'){
            config_prefix.append("/");
        }

        std::string config_path;

        spdlog::get(AIDB_DEBUG)->debug("backend:{}, model:{}", backend_lower, model_lower);


        if("onnx" == backend_lower){
#ifdef ENABLE_ORT
            config_path = config_prefix + "onnx_config.yaml";
#endif
        } else if("mnn" == backend_lower){
#ifdef ENABLE_MNN
            config_path = config_prefix + "mnn_config.yaml";
#endif
        } else if("ncnn" == backend_lower){
#ifdef ENABLE_NCNN
            config_path = config_prefix + "ncnn_config.yaml";
#endif
        } else if("tnn" == backend_lower){
#ifdef ENABLE_TNN
            config_path = config_prefix + "tnn_config.yaml";
#endif
        } else if("paddlelite" == backend_lower){
#ifdef ENABLE_PPLite
            config_path = config_prefix + "paddle_config.yaml";
#endif
        } else if("openvino" == backend_lower){
#ifdef ENABLE_OPV
            config_path = config_prefix + "openvino_config.yaml";
#endif
        } else{
            std::cout << "Not support backend:" << backend << std::endl;
            return nullptr;
        }
        if(config_path.empty()){
            std::cout << "remake to support backend:" << backend << std::endl;
            return nullptr;
        }

        auto node = YAML::LoadFile(config_path);
        auto AIDBZoo = node["AIDBZOO"];
        if(!AIDBZoo[model_lower.c_str()].IsDefined()){
            spdlog::get(AIDB_DEBUG)->error("model not support! backend:{}, model:{}", backend_lower, model_lower);
            return nullptr;
        }

        auto model_node = AIDBZoo[model_lower.c_str()];

        Engine* ptr_engine = nullptr;
        switch(engineType(model_node["backend"].as<std::string>())){

            case ONNX:{
#ifdef ENABLE_ORT
                ONNXParameter param = ONNXParameter(model_node);
                ptr_engine = new ONNXEngine();
                status = ptr_engine->init(param);
#endif
                break;
            }

            case MNN:{
#ifdef ENABLE_MNN
                MNNParameter param = MNNParameter(model_node);
                ptr_engine = new MNNEngine();
                status = ptr_engine->init(param);
#endif
                break;
            }

            case NCNN:{
#ifdef ENABLE_NCNN
                NCNNParameter param = NCNNParameter(model_node);
                ptr_engine = new NCNNEngine();
                status = ptr_engine->init(param);
#endif
                break;
            }
            case TNN:{
#ifdef ENABLE_TNN
                TNNParameter param = TNNParameter(model_node);
                ptr_engine = new TNNEngine();
                status = ptr_engine->init(param);
#endif
            }
                break;
            case OPENVINO:{
#ifdef ENABLE_OPV
                OPVParameter param = OPVParameter(model_node);
                ptr_engine = new OPVEngine();
                status = ptr_engine->init(param);
#endif
            }
                break;
            case PADDLE_LITE:{
#ifdef ENABLE_PPLite
                PPLiteParameter param = PPLiteParameter(model_node);
                ptr_engine = new PPLiteEngine();
                status = ptr_engine->init(param);
#endif
            }
                break;
            case TRT:
                break;
            default:
                break;
        }

        if(NO_ERROR != status){
            delete ptr_engine;
            spdlog::get(AIDB_DEBUG)->error("model init error! backend:{}, model:{}, status:{}", backend_lower, model_lower, status);
            return nullptr;
        }
        AIDBInput* ptr_input = nullptr;
        if(model_node["detail"]["PreProcess"].IsDefined()){
            auto input_node = model_node["detail"]["PreProcess"];
            ptr_input = new ImageInput(input_node);
        }

        return new Interpreter(ptr_engine, ptr_input);
    }
#ifdef ENABLE_NCNN_WASM
    Interpreter *
    Interpreter::createInstance(const void *buffer_in1, const void *buffer_in2, const std::string &config) {
        StatusCode status = NO_ERROR;

        aidb_log_init(AIDB_DEBUG, "debug");

        Engine* ptr_engine = nullptr;
        NCNNParameter param = NCNNParameter(config);
        ptr_engine = new NCNNEngine();
        status = ptr_engine->init(param, buffer_in1, buffer_in2);

        if(NO_ERROR != status){
            delete ptr_engine;
            spdlog::get(AIDB_DEBUG)->error("model init error!, status:{}", status);
            return nullptr;
        }

        AIDBInput* ptr_input = nullptr;

        ptr_input = new ImageInput(config);

        return new Interpreter(ptr_engine, ptr_input);
    }
#endif
    void Interpreter::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape){
        ENGINE_ASSERT(nullptr != _ptr_engine)
        _ptr_engine->forward(frame, frame_width, frame_height, frame_channel, outputs, outputs_shape);
    }

#ifndef ENABLE_NCNN_WASM


    cv::Mat Interpreter::operator<< (const cv::Mat &image){
        assert(nullptr != _ptr_input);
        cv::Mat blob;
        _ptr_input->forward(image, blob);
        return blob;
    }


    cv::Mat Interpreter::operator<< (const std::string &image_path){
        assert(nullptr != _ptr_input);
        auto bgr = cv::imread(image_path);
        assert(!bgr.empty());
        cv::Mat blob;
        _ptr_input->forward(bgr, blob);
        return blob;
    }
#else
//    void Interpreter::forward(const ncnn::Mat frame, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape){
//        ENGINE_ASSERT(nullptr != _ptr_engine)
//        _ptr_engine->forward(frame, outputs, outputs_shape);
//    }
//    void Interpreter::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape){
//        ENGINE_ASSERT(nullptr != _ptr_engine)
//        _ptr_engine->forward(frame, frame_width, frame_height, frame_channel, outputs, outputs_shape);
//    }

    ncnn::Mat Interpreter::operator<< (const cv::Mat &image){
        assert(nullptr != _ptr_input);
        ncnn::Mat blob;
        _ptr_input->forward(image, blob);
        return blob;
    }


    ncnn::Mat Interpreter::operator<< (const std::string &image_path) {
        assert(nullptr != _ptr_input);
        auto bgr = cv::imread(image_path);
        assert(!bgr.empty());
        ncnn::Mat blob;
        _ptr_input->forward(bgr, blob);
        return blob;
    }
#endif
    int Interpreter::width() const {
        return _ptr_input->width();
    }
    int Interpreter::height() const {
        return _ptr_input->height();
    }
    int Interpreter::channel() const {
        return _ptr_input->channel();
    }

    float Interpreter::scale_h() const {
        return _ptr_input->_scale_h;
    }

    float Interpreter::scale_w() const {
        return _ptr_input->_scale_w;
    }

//    Input &Input::operator>>(cv::Mat &dst) {
//        this->_ptr_input->_src_image.copyTo(dst);
//        return *this;
//    }
    void Interpreter::operator>>(cv::Mat &dst) {
//        this->_ptr_input->_src_image.copyTo(dst);
        dst = this->_ptr_input->_src_image.clone();
//        return *this;
    }

    void Interpreter::releaseInstance(Interpreter *ins) {
        if(ins){
            delete ins;
            ins = nullptr;
        }
    }

    void Interpreter::set_roi(cv::Rect2f roi) {
        _ptr_input->set_roi(true);
        _ptr_input->set_roi(roi);
    }


}