#include "backend/openvino/OPVEngine.hpp"
#include <iostream>

namespace AIDB {

    void printInputAndOutputsInfo(const ov::Model& network) {
        std::cout << "model name: " << network.get_friendly_name() << std::endl;

        const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
        for (const ov::Output<const ov::Node> input : inputs) {
            std::cout << "    inputs" << std::endl;

            const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
            std::cout << "        input name: " << name << std::endl;

            const ov::element::Type type = input.get_element_type();
            std::cout << "        input type: " << type << std::endl;

            const ov::Shape shape = input.get_shape();
            std::cout << "        input shape: " << shape << std::endl;
        }

        const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
        for (const ov::Output<const ov::Node> output : outputs) {
            std::cout << "    outputs" << std::endl;

            const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
            std::cout << "        output name: " << name << std::endl;

            const ov::element::Type type = output.get_element_type();
            std::cout << "        output type: " << type << std::endl;

            const ov::Shape shape = output.get_shape();
            std::cout << "        output shape: " << shape << std::endl;
        }
    }
    
    OPVEngine::OPVEngine() {
       _model = nullptr;
    }

    void OPVEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        // -------- Prepare input --------
        for (auto & _input_node : _input_nodes) {
            // just wrap image data by ov::Tensor without allocating of new memory

            ov::Shape input_dim(_input_node.second.begin(), _input_node.second.end());

            for(auto &dim: input_dim){
                if(dim == -1) dim = frame_height;
                if(dim == -2) dim = frame_width;
            }
            ov::Tensor input_tensor = ov::Tensor(_input_type, input_dim, (void*)frame);
            _infer_request.set_tensor(_input_node.first, input_tensor);

        }
        // -------- Do inference synchronously --------
        _infer_request.infer();

        // -------- Process output --------
        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());

        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];

            const ov::Tensor& output_tensor = _infer_request.get_tensor(name);

            auto dim = output_tensor.get_shape();
            for(int i =0; i < dim.size(); i++){
                outputs_shape[index].push_back(dim[i]);
//                std::cout << dim[i] << "-";
            }
//            std::cout << "\n";

            outputs[index].resize(output_tensor.get_size());
            ::memcpy(outputs[index].data(), output_tensor.data<float>(), output_tensor.get_size() * sizeof(float));
        }

    }


    OPVEngine::~OPVEngine() {

    }

    StatusCode OPVEngine::init(const Parameter &param) {
        _model_name = param._model_name;
        _backend_name = param._backend_name;
        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;
//        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());

        _model = _core.read_model(param._model_path, param._param_path);

        if (nullptr == _model) {
            spdlog::get(AIDB_DEBUG)->error("backend openvino read model failed!");
            return MODEL_CREATE_ERROR;
        }

        //printInputAndOutputsInfo(*_model);

        // -------- Loading a model to the device --------
        _compiled_model = _core.compile_model(_model, "CPU");

        // -------- Create an infer request --------
        _infer_request = _compiled_model.create_infer_request();

        spdlog::get(AIDB_DEBUG)->debug("backend openvino init succeed!");
        return NO_ERROR;

    }

    StatusCode OPVEngine::init(const Parameter &, const void *buffer_in1, const void *buffer_in2) {
        return NOT_IMPLEMENT;
    }

    void OPVEngine::forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape,
                            std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        assert(_input_nodes.size() == input.size() && input_shape.size() == input.size());


        // -------- Prepare input --------
        std::vector<char*> input_node_names;
        for(int i = 0; i < _input_node_name.size(); i++){
//        for (auto & _input_node : _input_nodes) {
            // just wrap image data by ov::Tensor without allocating of new memory
            auto _input_node = _input_nodes[_input_node_name[i]];

            ov::Shape input_dim(_input_node.begin(), _input_node.end());
            for(int d = 0; d < input_dim.size(); d++){
                if(-1 == input_dim[d]){
                    input_dim[d] = input_shape[i][d];
                }
            }

            ov::Tensor input_tensor = ov::Tensor(_input_type, input_dim, (void*)input[i]);
            _infer_request.set_tensor(_input_node_name[i], input_tensor);

        }

        // -------- Do inference synchronously --------
        _infer_request.infer();

        // -------- Process output --------
        outputs.resize(_output_node_name.size());
        outputs_shape.resize(_output_node_name.size());

        for(const auto& name: _output_node_name){
            int index = &name - &_output_node_name[0];

            const ov::Tensor& output_tensor = _infer_request.get_tensor(name);

            auto dim = output_tensor.get_shape();
            for(int i =0; i < dim.size(); i++){
                outputs_shape[index].push_back(dim[i]);
//                std::cout << dim[i] << "-";
            }
//            std::cout << "\n";

            outputs[index].resize(output_tensor.get_size());
            ::memcpy(outputs[index].data(), output_tensor.data<float>(), output_tensor.get_size() * sizeof(float));
        }

    }


}

