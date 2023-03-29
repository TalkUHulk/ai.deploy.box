#include "backend/onnxruntime/ONNXEngine.hpp"

namespace AIDB{

    ONNXEngine::ONNXEngine() {
        _session = nullptr;
    }

    ONNXEngine::~ONNXEngine(){
        if (nullptr != _session) {
            _session->release();
            _session = nullptr;
        }
    }

    void ONNXEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape){

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> ort_inputs;

        std::vector<char*> input_node_names;
        for (auto & _input_node : _input_nodes) {
            std::vector<int64_t> input_dim(_input_node.second.begin(), _input_node.second.end());
            for(auto &dim: input_dim){
                if(dim == -1) dim = frame_height;
                if(dim == -2) dim = frame_width;
            }

            Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info,
                                                               (void *) frame,
                                                               frame_width * frame_height * frame_channel * sizeof(float) ,
                                                              input_dim.data(),
                                                              input_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            ort_inputs.push_back(std::move(input_tensor));
            input_node_names.push_back(const_cast<char*>(_input_node.first.c_str()));
        }

        std::vector<Ort::Value> ort_outputs;
        std::vector<char*> output_node_names;

        for(const auto& name: _output_node_name){
            output_node_names.push_back(const_cast<char*>(name.c_str()));

        }

        ort_outputs = _session->Run(Ort::RunOptions{nullptr},
                                      input_node_names.data(),
                                      ort_inputs.data(),
                                      ort_inputs.size(),
                                      output_node_names.data(),
                                      output_node_names.size());

        outputs.clear();
        outputs.resize(_output_node_name.size());
        outputs_shape.clear();
        outputs_shape.resize(_output_node_name.size());

        for (auto &ort_output: ort_outputs){
            auto index = &ort_output - &ort_outputs[0];
            auto info = ort_output.GetTensorTypeAndShapeInfo();
            auto output_len = info.GetElementCount();
            auto dim_count = info.GetDimensionsCount();
            std::vector<int64_t> dims(dim_count, 1);
            info.GetDimensions(dims.data(), info.GetDimensionsCount());

            for (int cc = 0; cc < dim_count; cc++){
                outputs_shape[index].push_back(int(dims[cc]));
//                std::cout << "@@" << dims[cc];
            }
//            std::cout << "\n";
            ort_output.GetTensorData<float>();
            outputs[index].resize(output_len);
            ::memcpy(outputs[index].data(), ort_output.GetTensorData<float>(), sizeof(float)*output_len);

        }

    }

    StatusCode ONNXEngine::init(const Parameter &param){
        _model_name = param._model_name;
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
        _input_nodes = param._input_nodes;

        _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, _model_name.c_str());

        // cuda
        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);

        _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        _session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);

        _session = std::make_shared<Ort::Session>(_env, param._model_path.c_str(), _session_options);

        if (nullptr == _session) {
            spdlog::get(AIDB_DEBUG)->error("backend onnx init failed!");
            return MODEL_CREATE_ERROR;
        }
        spdlog::get(AIDB_DEBUG)->debug("backend onnx init succeed!");
        return NO_ERROR;
    }

    StatusCode ONNXEngine::init(const Parameter &, const void *buffer_in1, const void *buffer_in2) {
        return NOT_IMPLEMENT;
    }

//    StatusCode ONNXEngine::init(const Parameter &param, const uint8_t *buffer_in, size_t buffer_size_in){
//        _model_name = param._model_name;
//        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());
//        _input_nodes = param._input_nodes;
//
//        if (nullptr == buffer_in) {
//            return INPUT_DATA_ERROR;
//        }
//        if (0 == buffer_size_in) {
//            return INPUT_DATA_ERROR;
//        }
//
//        _env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, _model_name.c_str());
//
//        _session_options.SetIntraOpNumThreads(param._numThread);
//
//        // cuda
//        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
//
//        _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//        _session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
//
//        _session = std::make_shared<Ort::Session>(_env, buffer_in, buffer_size_in, _session_options);
//
//        if (nullptr == _session) {
//            return MODEL_CREATE_ERROR;
//        }
//
//        return NO_ERROR;
//    }
}