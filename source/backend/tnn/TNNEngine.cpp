#include "backend/tnn/TNNEngine.hpp"
#include <iostream>
#include <fstream>
#include <numeric>


namespace AIDB {
    TNNEngine::TNNEngine() {
        _net = nullptr;
        _instance = nullptr;
    }

    void TNNEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {


        for (auto & _input_node : _input_nodes) {

            std::vector<int> input_dim(_input_node.second.begin(), _input_node.second.end());

            if(_dynamic) {

                for (auto &dim: input_dim) {
                    if (dim == -1) dim = frame_height;
                    if (dim == -2) dim = frame_width;
                }
                _instance->Reshape({{_input_node.first, input_dim}});
            }
            auto input_mat = std::make_shared<tnn::Mat>(_input_device_type, tnn::NCHW_FLOAT,
                                                        input_dim, (void *) frame);
            auto input_cvt_param = TNN_NS::MatConvertParam();
            auto status_in = _instance->SetInputMat(input_mat, input_cvt_param, _input_node.first);
            if (status_in != tnn::TNN_OK) {
                std::cout << status_in.description().c_str() << "\n";
                throw std::runtime_error("SetInputMat Error");
            }
        }


        auto status = _instance->Forward();

        if (status != tnn::TNN_OK){
            std::cout << status.description().c_str() << "\n";
            throw std::runtime_error("Forward Error");
        }

        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());
        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];

            tnn::MatConvertParam cvt_param;
            std::shared_ptr<tnn::Mat> output_mat;
            auto status_out = _instance->GetOutputMat(output_mat, cvt_param, name, _output_device_type);

            if (status_out != tnn::TNN_OK){
                std::cout << status_out.description().c_str() << "\n";
                throw std::runtime_error("GetOutputMat Error");
            }
            auto output_dims = output_mat->GetDims();

            for(auto dim: output_dims){
                outputs_shape[index].push_back(dim);
//                std::cout << dim << "、";
            }
            int output_len = std::accumulate(outputs_shape[index].begin(), outputs_shape[index].end(), 1, std::multiplies<int>());
//            std::cout << "\n";
            outputs[index].resize(output_len);
            ::memcpy(outputs[index].data(), output_mat->GetData(), output_len * sizeof(float));
        }

    }


    TNNEngine::~TNNEngine() {
        _net = nullptr;
        _instance = nullptr;
    }

    inline tnn::DimsVector TNNEngine::get_input_shape(const std::string& name){
        return TNNEngine::get_input_shape(_instance, name);
    }

    inline tnn::DimsVector TNNEngine::get_output_shape(const std::string& name){
        return TNNEngine::get_output_shape(_instance, name);
    }

    inline std::vector<std::string> TNNEngine::get_input_names(){
        return TNNEngine::get_input_names(_instance);
    }

    inline std::vector<std::string> TNNEngine::get_output_names(){
        return TNNEngine::get_output_names(_instance);
    }

    inline tnn::MatType TNNEngine::get_output_mat_type(const std::string& name){
        return TNNEngine::get_output_mat_type(_instance, name);
    }

    inline tnn::DataFormat TNNEngine::get_output_data_format(const std::string& name){
        return TNNEngine::get_output_data_format(_instance, name);
    }

    inline tnn::MatType TNNEngine::get_input_mat_type(const std::string& name){
        return TNNEngine::get_input_mat_type(_instance, name);
    }

    inline tnn::DataFormat TNNEngine::get_input_data_format(const std::string& name){
        return TNNEngine::get_input_data_format(_instance, name);
    }
    
    StatusCode TNNEngine::init(const Parameter &param) {
        _model_name = param._model_name;
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;
//        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());
        _dynamic = param._dynamic;

        LOGD("==@@##!!", "_param_path: %s", param._param_path.c_str());
        LOGD("==@@##!!", "_model_path: %s", param._model_path.c_str());

        std::string proto_content_buffer, model_content_buffer;
        proto_content_buffer = content_buffer_from(param._param_path.c_str());
        model_content_buffer = content_buffer_from(param._model_path.c_str());

        tnn::ModelConfig model_config;
        model_config.model_type = tnn::MODEL_TYPE_TNN;
        model_config.params = {proto_content_buffer, model_content_buffer};

        // init TNN net
        tnn::Status status;
        _net = std::make_shared<tnn::TNN>();
        status = _net->Init(model_config);

        if (status != tnn::TNN_OK || !_net){
            spdlog::get(AIDB_DEBUG)->error("backend tnn init failed! status:{}", status.description());
            return MODEL_CREATE_ERROR;
        }

        // init instance
        tnn::NetworkConfig network_config;
        network_config.library_path = {""};
        network_config.device_type = _network_device_type;

        _instance = _net->CreateInst(network_config, status);

        if (status != tnn::TNN_OK || !_instance){
            spdlog::get(AIDB_DEBUG)->error("backend tnn CreateInst failed! status:{}", status.description());
            return MODEL_CREATE_ERROR;
        }

        _instance->SetCpuNumThreads(param._numThread);
        spdlog::get(AIDB_DEBUG)->debug("backend tnn init succeed!");
        return NO_ERROR;

    }


    std::string TNNEngine::content_buffer_from(const char *proto_or_model_path){

        std::ifstream file(proto_or_model_path, std::ios::binary);
        if (file.is_open())
        {
            file.seekg(0, file.end);
            int size = file.tellg();
            char *content = new char[size];
            file.seekg(0, file.beg);
            file.read(content, size);
            std::string file_content;
            file_content.assign(content, size);
            delete[] content;
            file.close();
            return file_content;
        } // empty buffer
        else
        {
            std::cout << "Can not open " << proto_or_model_path << "\n";
            return "";
        }
    }

    
    tnn::DimsVector TNNEngine::get_input_shape(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_){

        tnn::DimsVector shape = {};
        tnn::BlobMap blob_map = {};
        if (instance_) instance_->GetAllInputBlobs(blob_map);

        if (name_.empty() && !blob_map.empty())
            if (blob_map.begin()->second)
                shape = blob_map.begin()->second->GetBlobDesc().dims;

        if (blob_map.find(name_) != blob_map.end() && blob_map[name_]){
            shape = blob_map[name_]->GetBlobDesc().dims;
        }

        return shape;
    }


    tnn::DimsVector TNNEngine::get_output_shape(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_)
    {
        tnn::DimsVector shape = {};
        tnn::BlobMap blob_map = {};
        if (instance_) instance_->GetAllOutputBlobs(blob_map);


        if (name_.empty() && !blob_map.empty())
            if (blob_map.begin()->second)
                shape = blob_map.begin()->second->GetBlobDesc().dims;

        if (blob_map.find(name_) != blob_map.end() && blob_map[name_]) {
            shape = blob_map[name_]->GetBlobDesc().dims;
        }

        return shape;
    }


    std::vector<std::string> TNNEngine::get_input_names(const std::shared_ptr<tnn::Instance> &instance_) {
        std::vector<std::string> names;
        if (instance_){
            tnn::BlobMap blob_map;
            instance_->GetAllInputBlobs(blob_map);
            for (const auto &item : blob_map)
            {
                names.push_back(item.first);
            }
        }
        return names;
    }


    std::vector<std::string> TNNEngine::get_output_names(const std::shared_ptr<tnn::Instance> &instance_){
        std::vector<std::string> names;
        if (instance_){
            tnn::BlobMap blob_map;
            instance_->GetAllOutputBlobs(blob_map);
            for (const auto &item : blob_map)
            {
                names.push_back(item.first);
            }
        }
        return names;
    }


    tnn::MatType TNNEngine::get_output_mat_type(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_){
        if (instance_){
            tnn::BlobMap output_blobs;
            instance_->GetAllOutputBlobs(output_blobs);
            auto blob = (name_.empty()) ? output_blobs.begin()->second : output_blobs[name_];
            if (blob->GetBlobDesc().data_type == tnn::DATA_TYPE_INT32)
            {
                return tnn::NC_INT32;
            }
        }
        return tnn::NCHW_FLOAT;
    }


    tnn::DataFormat TNNEngine::get_output_data_format(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_)
    {
        if (instance_){
            tnn::BlobMap output_blobs;
            instance_->GetAllOutputBlobs(output_blobs);
            auto blob = (name_.empty()) ? output_blobs.begin()->second : output_blobs[name_];
            return blob->GetBlobDesc().data_format;
        }
        return tnn::DATA_FORMAT_NCHW;
    }


    tnn::MatType TNNEngine::get_input_mat_type(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_){
        if (instance_){
            tnn::BlobMap input_blobs;
            instance_->GetAllInputBlobs(input_blobs);
            auto blob = (name_.empty()) ? input_blobs.begin()->second : input_blobs[name_];
            if (blob->GetBlobDesc().data_type == tnn::DATA_TYPE_INT32)
            {
                return tnn::NC_INT32;
            }
        }
        return tnn::NCHW_FLOAT;
    }


    tnn::DataFormat TNNEngine::get_input_data_format(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_){
        if (instance_){
            tnn::BlobMap input_blobs;
            instance_->GetAllInputBlobs(input_blobs);
            auto blob = (name_.empty()) ? input_blobs.begin()->second : input_blobs[name_];
            return blob->GetBlobDesc().data_format;
        }
        return tnn::DATA_FORMAT_NCHW;
    }

    StatusCode TNNEngine::init(const Parameter &, const void *buffer_in1, const void *buffer_in2) {
        return NOT_IMPLEMENT;
    }


}

