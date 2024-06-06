#include "backend/ncnn/NCNNEngine.hpp"
#include <iostream>
#include "CustomLayer.hpp"
#include <numeric>

namespace AIDB {

#define NCNN_LAYER_CREATOR(name)  name##_layer_creator

    NCNNEngine::NCNNEngine() {
        _net = std::make_shared<ncnn::Net>();
    }


    void NCNNEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        ncnn::Extractor ex = _net->create_extractor();
        for (auto & _input_node : _input_nodes) {
            ncnn::Mat in(frame_width, frame_height, frame_channel);
            memcpy(in.data, frame, sizeof(float) * frame_width * frame_height * frame_channel);

//            ncnn::Mat in = ncnn::Mat::from_pixels_resize(reinterpret_cast<const unsigned char *>(frame), ncnn::Mat::PIXEL_RGBA2RGB, frame_width, frame_height, 640, 640);

            //            auto bgr = cv::imread("coco2.jpg");
//            cv::resize(bgr, bgr, cv::Size(640, 640));
//            ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data,  ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);

//            ex.input(_input_node.first.c_str(), in);
//            ex.input(std::stoi(_input_node.first), in);
#ifndef ENABLE_NCNN_WASM
            ex.input(_input_node.first.c_str(), in);
#else
            ex.input(std::stoi(_input_node.first), in);
#endif
        }
        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());
        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];
            ncnn::Mat pred;

//            ex.extract(name.c_str(), pred);
//            ex.extract(std::stoi(name), pred);
#ifndef ENABLE_NCNN_WASM
            ex.extract(name.c_str(), pred); //mobilesam floating point exception
#else
            ex.extract(std::stoi(name), pred);
#endif
            outputs_shape[index].push_back(1); //占位，方便统一后处理
//            std::cout << "w:" << pred.w << "h:" << pred.h << "c" << pred.c << "d" << pred.d<< " dims:"<< pred.dims << std::endl;
            if (pred.dims == 1){
                outputs_shape[index].push_back(pred.w);
            }
            if (pred.dims == 2){
                outputs_shape[index].push_back(pred.h);
                outputs_shape[index].push_back(pred.w);
            }
            if (pred.dims == 3){
                outputs_shape[index].push_back(pred.c);
                outputs_shape[index].push_back(pred.h);
                outputs_shape[index].push_back(pred.w);
            }
            if (pred.dims == 4){
                outputs_shape[index].push_back(pred.c);
                outputs_shape[index].push_back(pred.d);
                outputs_shape[index].push_back(pred.h);
                outputs_shape[index].push_back(pred.w);
            }
            outputs[index].resize(pred.total());
            ::memcpy(outputs[index].data(), pred.data, pred.total() * sizeof(float));
        }

    }

    NCNNEngine::~NCNNEngine() {
        if(nullptr != _net){
            _net->clear();
            _net = nullptr;
        }

    }

    StatusCode NCNNEngine::init(const Parameter &param) {
        _model_name = param._model_name;
        _backend_name = param._backend_name;
//        _opt.num_threads = param._numThread;
//        _opt.blob_allocator = &_ncnn_blob_pool_allocator;
//        _opt.workspace_allocator = &_ncnn_workspace_pool_allocator;

        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;
//        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());

        for(const auto& register_layer: param._register_layers){
            _net->register_custom_layer(register_layer.c_str(), custom_layer_map[register_layer.c_str()]);
        }

        auto status = _net->load_param(param._param_path.c_str());
        if (0 != status) {
            return MODEL_CREATE_ERROR;
        }

        status = _net->load_model(param._model_path.c_str());
        if (0 != status) {
            return MODEL_CREATE_ERROR;
        }

//        _opt.lightmode = false;
//        _opt.use_bf16_storage = false;
//        _opt.use_fp16_arithmetic = false;
//        _opt.use_fp16_packed = false;
//        _opt.use_fp16_storage = true;
//        _opt.use_fp16_arithmetic = true;
//        _opt.use_int8_packed = false;
//        _opt.use_int8_storage = false;
//        _opt.use_int8_arithmetic = false;

        _net->opt = _opt;
        spdlog::get(AIDB_DEBUG)->debug("backend ncnn init succeed!");
        return NO_ERROR;

    }

//    StatusCode NCNNEngine::init(const Parameter &, const uint8_t *buffer_in, size_t buffer_size_in) {
//        return NOT_IMPLEMENT;
//    }

    StatusCode NCNNEngine::init(const Parameter& param, const void *buffer_in1, const void* buffer_in2)  {
        _model_name = param._model_name;
        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;

        for(const auto& register_layer: param._register_layers){
            _net->register_custom_layer(register_layer.c_str(), custom_layer_map[register_layer.c_str()]);
        }

//        Most loading functions return 0 if success, except loading alexnet.param.bin and alexnet.bin from file memory, which returns the bytes consumed after loading

        auto status = _net->load_param((uint8_t*)buffer_in1);
//        if (0 != status) {
//            spdlog::get(AIDB_DEBUG)->error("backend ncnn load param failed! status:{}", status);
//            return MODEL_CREATE_ERROR;
//        }

        status = _net->load_model((uint8_t*)buffer_in2);
//        if (0 != status) {
//            spdlog::get(AIDB_DEBUG)->error("backend ncnn load model failed! status:{}", status);
//            return MODEL_CREATE_ERROR;
//        }

        _net->opt = _opt;
        spdlog::get(AIDB_DEBUG)->debug("backend ncnn init succeed!");
        return NO_ERROR;
    }

    void NCNNEngine::forward(const std::vector<void *> &input, const std::vector<std::vector<int>> &input_shape,
                             std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        assert(_input_nodes.size() == input.size() && input_shape.size() == input.size());

        ncnn::Extractor ex = _net->create_extractor();
        std::vector<char*> input_node_names;
        for(int i = 0; i < _input_node_name.size(); i++){
//        for (auto & _input_node : _input_nodes) {
            auto _input_node = _input_nodes[_input_node_name[i]];
            std::vector<int64_t> input_dim(_input_node.begin(), _input_node.end());

            for(int d = 0; d < input_dim.size(); d++){
                if(-1 == input_dim[d]){
                    input_dim[d] = input_shape[i][d];
                }
            }
            ncnn::Mat in;
            size_t _elemsize = 4;
            switch (input_dim.size()) {
                case 1:{
                    in = ncnn::Mat((int)input_dim[0], _elemsize);
                    break;
                }
                case 2:{
                    in = ncnn::Mat((int)input_dim[1], _elemsize);
                    break;
                }
                case 3:{
                    in = ncnn::Mat((int)input_dim[2], (int)input_dim[1], _elemsize);
                    break;
                }
                case 4:{
                    in = ncnn::Mat((int)input_dim[3], (int)input_dim[2], (int)input_dim[1], _elemsize);
                    break;
                }

            }
            // ncnn not support batch
            memcpy(in.data, input[i], sizeof(float) * std::accumulate(
                    input_shape[i].begin(),
                    input_shape[i].end(),
                    1,
                    std::multiplies<int>()
            ));

//            ncnn::Mat in = ncnn::Mat::from_pixels_resize(reinterpret_cast<const unsigned char *>(frame), ncnn::Mat::PIXEL_RGBA2RGB, frame_width, frame_height, 640, 640);

            //            auto bgr = cv::imread("coco2.jpg");
//            cv::resize(bgr, bgr, cv::Size(640, 640));
//            ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data,  ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);

//            ex.input(_input_node.first.c_str(), in);
//            ex.input(std::stoi(_input_node.first), in);
#ifndef ENABLE_NCNN_WASM
            ex.input(_input_node_name[i].c_str(), in);
//            std::cout << _input_node_name[i] << ":" << "w=" << in.w << ", h=" << in.h << ", c=" << in.c << ", dim=" << in.dims << std::endl;
#else
            ex.input(std::stoi(_input_node.first), in);
#endif
        }

        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());
        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];

            ncnn::Mat pred;

#ifndef ENABLE_NCNN_WASM
            ex.extract(name.c_str(), pred);
#else
            ex.extract(std::stoi(name), pred);
#endif
            outputs_shape[index].clear();
            outputs_shape[index].push_back(1); //占位，方便统一后处理
            if (pred.dims == 1){
                outputs_shape[index].push_back(pred.w);
            }
            if (pred.dims == 2){
                outputs_shape[index].push_back(pred.h);
                outputs_shape[index].push_back(pred.w);
            }
            if (pred.dims == 3){
                outputs_shape[index].push_back(pred.c);
                outputs_shape[index].push_back(pred.h);
                outputs_shape[index].push_back(pred.w);
            }
            if (pred.dims == 4){
                outputs_shape[index].push_back(pred.d);
                outputs_shape[index].push_back(pred.c);
                outputs_shape[index].push_back(pred.h);
                outputs_shape[index].push_back(pred.w);
            }

//            outputs_size.push_back(pred.total());
            outputs[index].resize(pred.total());
            ::memcpy(outputs[index].data(), pred.data, pred.total() * sizeof(float));
        }

    }



}

