#include "backend/mnn/MNNEngine.hpp"
#include <iostream>
#include <numeric>


namespace AIDB {
    MNNEngine::MNNEngine() {
        _mnn_session = nullptr;
        _mnn_net = nullptr;
    }

    void MNNEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        for (auto & _input_node : _input_nodes) {

            MNN::Tensor *input_tensor = _input_node.first.empty()? get_input_tensor(): get_input_tensor(_input_node.first.c_str());
            if(_dynamic){

                std::vector<int> input_dim(_input_node.second.begin(), _input_node.second.end());
                for(auto &dim: input_dim){
                    if(dim == -1) dim = frame_height;
                    if(dim == -2) dim = frame_width;
                }
                _mnn_net->resizeTensor(input_tensor, input_dim);
                _mnn_net->resizeSession(_mnn_session);
            }

            auto nhwc_Tensor = MNN::Tensor::create<float>(input_tensor->shape(), nullptr, MNN::Tensor::CAFFE);
            auto nchw_data   = nhwc_Tensor->host<float>();
            auto nchw_size   = nhwc_Tensor->size();
            ::memcpy(nchw_data, frame, nchw_size);
            input_tensor->copyFromHostTensor(nhwc_Tensor);


        }

        _mnn_net->runSession(_mnn_session);

        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());

        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];
            std::shared_ptr<MNN::Tensor> tensor = get_output_by_name(name.c_str());
            for(auto dim: tensor->shape()){
//                std::cout << dim << ",";
                outputs_shape[index].push_back(dim);
            }
//
//            std::cout << "\n";
            outputs[index].resize(tensor->size());
            ::memcpy(outputs[index].data(), tensor->host<float>(), tensor->size());
        }
    }


    MNNEngine::~MNNEngine() {
        if (nullptr != _mnn_net) {
            _mnn_net->releaseModel();
            _mnn_net->releaseSession(_mnn_session);
        }
    }

    StatusCode MNNEngine::init(const Parameter &param) {
        _model_name = param._model_name;
        _backend_name = param._backend_name;
        _net_cfg.type = MNN_FORWARD_CPU;//MNN_FORWARD_VULKAN;//MNN_FORWARD_VULKAN;//MNN_FORWARD_CPU;//MNN_FORWARD_OPENCL;
        _net_cfg.numThread = param._numThread;
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;
        _dynamic = param._dynamic;
//        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());

        _mnn_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(param._model_path.c_str()));
        if (nullptr == _mnn_net) {
            spdlog::get(AIDB_DEBUG)->error("backend mnn create interpreter failed!");
            return MODEL_CREATE_ERROR;
        }

        _mnn_session = _mnn_net->createSession(_net_cfg);

        if (nullptr == _mnn_session) {
            spdlog::get(AIDB_DEBUG)->error("backend mnn create session failed!");
            return SESSION_CREATE_ERROR;
        }

        //this->reshape_input(param._input_shape);
        spdlog::get(AIDB_DEBUG)->debug("backend mnn init succeed!");
        return NO_ERROR;

    }

//    StatusCode MNNEngine::init(const Parameter &param, const uint8_t *buffer_in, size_t buffer_size_in) {
//        _model_name = param._model_name;
//        _net_cfg.type = MNN_FORWARD_CPU;
//        _net_cfg.numThread = param._numThread;
//        _output_node_name = param._output_node_name;
////        _input_node_name = param._input_node_name;
//        _input_nodes = param._input_nodes;
//        _dynamic = param._dynamic;
//
//        if (nullptr == buffer_in) {
//            return INPUT_DATA_ERROR;
//        }
//        if (0 == buffer_size_in) {
//            return INPUT_DATA_ERROR;
//        }
//        _mnn_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(buffer_in, buffer_size_in));
//        _mnn_session = _mnn_net->createSession(_net_cfg);
//        if (nullptr == _mnn_session) {
//            return SESSION_CREATE_ERROR;
//        }
//
//        //this->reshape_input(param._input_shape);
//        return NO_ERROR;
//    }

    void MNNEngine::reshape_input(const std::vector<int>& dim) {
        MNN::Tensor *input;
        input = _mnn_net->getSessionInput(_mnn_session, nullptr);
        _mnn_net->resizeTensor(input, dim);
        _mnn_net->resizeSession(_mnn_session);

    }

    MNN::Tensor *MNNEngine::get_input_tensor() {
        MNN::Tensor *input;
        input = _mnn_net->getSessionInput(_mnn_session, nullptr);
        return input;
    }

    MNN::Tensor *MNNEngine::get_input_tensor(const char *node_name) {
        MNN::Tensor *input;
        input = _mnn_net->getSessionInput(_mnn_session, node_name);
        return input;
    }

    std::shared_ptr<MNN::Tensor> MNNEngine::get_output_by_name(const char *name) {
        MNN::Tensor *out;
        out = _mnn_net->getSessionOutput(_mnn_session, name);
        std::shared_ptr<MNN::Tensor> res_tensor(new MNN::Tensor(out, MNN::Tensor::CAFFE));

        out->copyToHostTensor(res_tensor.get());
        return res_tensor;
    }

    StatusCode MNNEngine::init(const Parameter &, const void *buffer_in1, const void *buffer_in2) {
        return NOT_IMPLEMENT;
    }

    void MNNEngine::forward(std::vector<const void*> input, int frame_width, int frame_height, int frame_channel,
                            std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

//        assert(input.size() == _input_nodes.size());
//
//        for (int i = 0; i < _input_nodes.size(); ++i) {
////        for (auto & _input_node : _input_nodes) {
//            auto _input_node = _input_nodes[i];
//            MNN::Tensor *input_tensor = _input_node.first.empty()? get_input_tensor(): get_input_tensor(_input_node.first.c_str());
//            if(_dynamic){
//
//                std::vector<int> input_dim(_input_node.second.begin(), _input_node.second.end());
//                for(auto &dim: input_dim){
//                    if(dim == -1) dim = frame_height;
//                    if(dim == -2) dim = frame_width;
//                }
//                _mnn_net->resizeTensor(input_tensor, input_dim);
//                _mnn_net->resizeSession(_mnn_session);
//            }
//
//            auto nhwc_Tensor = MNN::Tensor::create<float>(input_tensor->shape(), nullptr, MNN::Tensor::CAFFE);
//            auto nchw_data   = nhwc_Tensor->host<float>();
//            auto nchw_size   = nhwc_Tensor->size();
//            ::memcpy(nchw_data, frame, nchw_size);
//            input_tensor->copyFromHostTensor(nhwc_Tensor);
//
//
//        }
//
//        _mnn_net->runSession(_mnn_session);
//
//        outputs.resize(_output_node_name.size());
//        outputs_shape.resize(_output_node_name.size());
//
//        for(const auto& name: _output_node_name){
//            int index = &name - &_output_node_name[0];
//            std::shared_ptr<MNN::Tensor> tensor = get_output_by_name(name.c_str());
//            for(auto dim: tensor->shape()){
////                std::cout << dim << ",";
//                outputs_shape[index].push_back(dim);
//            }
////
////            std::cout << "\n";
//            outputs[index].resize(tensor->size());
//            ::memcpy(outputs[index].data(), tensor->host<float>(), tensor->size());
//        }

    }
}

