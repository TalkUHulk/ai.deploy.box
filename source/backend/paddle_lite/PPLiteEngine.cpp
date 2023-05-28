#include "backend/paddle_lite/PPLiteEngine.hpp"
#include <iostream>
#include "utility/Logging.h"

namespace AIDB {

    int64_t ShapeProduction(const paddle::lite_api::shape_t& shape) {
        int64_t res = 1;
        for (auto i : shape) res *= i;
        return res;
    }

    PPLiteEngine::PPLiteEngine() {
       _predictor = nullptr;
    }

    void PPLiteEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        LOGD("====>>>", "forward");
        // prepare data
        for (auto & _input_node : _input_nodes) {
            std::vector<int64_t> input_dim(_input_node.second.begin(), _input_node.second.end());
            for(auto &dim: input_dim){
                if(dim == -1) dim = frame_height;
                if(dim == -2) dim = frame_width;
            }
            std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(_predictor->GetInputByName(_input_node.first)));
            input_tensor->Resize(input_dim);
            auto* data = input_tensor->mutable_data<float>();

            memcpy(data, frame, frame_channel * frame_width * frame_height * sizeof(float));
//            for (int i = 0; i < frame_channel * frame_width * frame_height; ++i) {
//                data[i] = frame[i];
//            }
        }

        // inference
        _predictor->Run();
        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());

        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];

            std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
                    std::move(_predictor->GetOutput(index)));
            // 转化为数据
            auto output_data=output_tensor->data<float>();

            auto elem_num = ShapeProduction(output_tensor->shape());
            for(auto dim: output_tensor->shape()){
                outputs_shape[index].push_back(dim);
//                std::cout << dim << ".";
            }
//            std::cout << "\n";
//            outputs_size.push_back(elem_num);
            outputs[index].resize(elem_num);

            ::memcpy(outputs[index].data(), output_data, elem_num * sizeof(float));
        }
    }


    PPLiteEngine::~PPLiteEngine() {

    }

    StatusCode PPLiteEngine::init(const Parameter &param) {
        _model_name = param._model_name;

        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;
//        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());

//        _config.set_model_from_file(param._model_path);

        _config.set_model_from_file(param._model_path);
        _config.set_threads(param._numThread);
        _predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(_config);

        spdlog::get(AIDB_DEBUG)->debug("backend paddle-lite init succeed!");

        return NO_ERROR;

    }

    StatusCode PPLiteEngine::init(const Parameter &, const void *buffer_in1, const void *buffer_in2) {
        return NOT_IMPLEMENT;
    }


}

