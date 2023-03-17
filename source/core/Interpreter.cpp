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

namespace AIDB {

//    Interpreter *Interpreter::createFromNode(const YAML::Node& engine_mode) {
//
//        StatusCode status = NO_ERROR;
//        Engine* ptr_engine = nullptr;
//        switch(engineType(engine_mode["backend"].as<std::string>())){
//
//            case ONNX:{
//#ifdef ENABLE_ORT
//                ONNXParameter param = ONNXParameter(engine_mode);
//                ptr_engine = new ONNXEngine();
//                status = ptr_engine->init(param);
//#endif
//                break;
//            }
//
//            case MNN:{
//#ifdef ENABLE_MNN
//                MNNParameter param = MNNParameter(engine_mode);
//                ptr_engine = new MNNEngine();
//                status = ptr_engine->init(param);
//#endif
//                break;
//            }
//
//            case NCNN:{
//#ifdef ENABLE_NCNN
//                NCNNParameter param = NCNNParameter(engine_mode);
//                ptr_engine = new NCNNEngine();
//                status = ptr_engine->init(param);
//#endif
//                break;
//            }
//            case TNN:{
//#ifdef ENABLE_TNN
//                TNNParameter param = TNNParameter(engine_mode);
//                ptr_engine = new TNNEngine();
//                status = ptr_engine->init(param);
//#endif
//            }
//                break;
//            case OPENVINO:{
//#ifdef ENABLE_OPV
//                OPVParameter param = OPVParameter(engine_mode);
//                ptr_engine = new OPVEngine();
//                status = ptr_engine->init(param);
//                std::cout << "init OPENVINO!\n";
//#endif
//            }
//                break;
//            case PADDLE_LITE:{
//#ifdef ENABLE_PPLite
//                PPLiteParameter param = PPLiteParameter(engine_mode);
//                ptr_engine = new PPLiteEngine();
//                status = ptr_engine->init(param);
//#endif
//            }
//                break;
//            case TRT:
//                break;
//            case TRITON:
//                break;
//            default:
//                std::cout << "default"  << std::endl;
//                break;
//        }
//
//        if(NO_ERROR != status){
//            delete ptr_engine;
//            std::cout << "!!! init error:" << status << std::endl;
//            return nullptr;
//        }
//        return new Interpreter(ptr_engine);
//    }

    void Interpreter::forward(const float *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape){
        ENGINE_ASSERT(nullptr != _ptr_engine)
        _ptr_engine->forward(frame, frame_width, frame_height, frame_channel, outputs, outputs_shape);
    }

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
        std::cout << "backend_lower:" << backend_lower << "; model_lower:"<< model_lower << std::endl;
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
            std::cout << "Not support Model:" << model << std::endl;
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
                std::cout << "init OPENVINO!\n";
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
            case TRITON:
                break;
            default:
                std::cout << "default"  << std::endl;
                break;
        }

        if(NO_ERROR != status){
            delete ptr_engine;
            std::cout << "!!! init error:" << status << std::endl;
            return nullptr;
        }
        AIDBInput* ptr_input = nullptr;
        if(model_node["detail"]["PreProcess"].IsDefined()){
            auto input_node = model_node["detail"]["PreProcess"];
            ptr_input = new ImageInput(input_node);
        }

        return new Interpreter(ptr_engine, ptr_input);
    }

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
        this->_ptr_input->_src_image.copyTo(dst);
//        return *this;
    }

    void Interpreter::releaseInstance(Interpreter *ins) {
        if(ins){
            delete ins;
            ins = nullptr;
        }
    }

}