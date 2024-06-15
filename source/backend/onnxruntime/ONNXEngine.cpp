#include "backend/onnxruntime/ONNXEngine.hpp"
#include <numeric>
#include <algorithm>
#ifdef SD_ON
#include "onnx.pb.h"
#include <fstream>
#endif

namespace AIDB{


    typedef std::uint64_t hash_t;

    constexpr hash_t prime = 0x100000001B3ull;
    constexpr hash_t basis = 0xCBF29CE484222325ull;

    hash_t hash_(char const* str)
    {
        hash_t ret{basis};

        while(*str){
            ret ^= *str;
            ret *= prime;
            str++;
        }

        return ret;
    }

    constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis)
    {
        return *str ? hash_compile_time(str+1, (*str ^ last_value) * prime) : last_value;
    }

    constexpr unsigned long long operator "" _hash(char const* p, size_t)
    {
        return hash_compile_time(p);
    }


    typedef enum AIDBElementDataType {
        AIDB_DATA_TYPE_UNDEFINED,
        AIDB_DATA_TYPE_FLOAT,   // maps to c type float
        AIDB_DATA_TYPE_UINT8,   // maps to c type uint8_t
        AIDB_DATA_TYPE_INT8,    // maps to c type int8_t
        AIDB_DATA_TYPE_UINT16,  // maps to c type uint16_t
        AIDB_DATA_TYPE_INT16,   // maps to c type int16_t
        AIDB_DATA_TYPE_INT32,   // maps to c type int32_t
        AIDB_DATA_TYPE_INT64,   // maps to c type int64_t
        AIDB_DATA_TYPE_STRING,  // maps to c++ type std::string
        AIDB_DATA_TYPE_BOOL,
        AIDB_DATA_TYPE_FLOAT16,
        AIDB_DATA_TYPE_DOUBLE,      // maps to c type double
        AIDB_DATA_TYPE_UINT32,      // maps to c type uint32_t
        AIDB_DATA_TYPE_UINT64,      // maps to c type uint64_t
    } AIDBElementDataType;

    AIDBElementDataType dataTypeFromStr(char const* str){
        switch(hash_(str)){
            case "i32"_hash:
                return AIDB_DATA_TYPE_INT32;
            case "i64"_hash:
                return AIDB_DATA_TYPE_INT64;
            case "f32"_hash:
                return AIDB_DATA_TYPE_FLOAT;
            case "f16"_hash:
                return AIDB_DATA_TYPE_FLOAT16;
            case "u8"_hash:
                return AIDB_DATA_TYPE_UINT8;
            case "i16"_hash:
                return AIDB_DATA_TYPE_INT16;
            case "i8"_hash:
                return AIDB_DATA_TYPE_INT8;
            case "f64"_hash:
                return AIDB_DATA_TYPE_DOUBLE;
            case "u16"_hash:
                return AIDB_DATA_TYPE_UINT16;
            case "u32"_hash:
                return AIDB_DATA_TYPE_UINT32;
            case "u64"_hash:
                return AIDB_DATA_TYPE_UINT64;
            default:
                return AIDB_DATA_TYPE_UNDEFINED;
        }

    }

    int sizeofA(AIDBElementDataType type){
        switch(type){
            case AIDB_DATA_TYPE_INT32:
                return sizeof(int);
            case AIDB_DATA_TYPE_FLOAT:
                return sizeof(float);
            case AIDB_DATA_TYPE_FLOAT16:
                return sizeof(float) / 2;
            case AIDB_DATA_TYPE_UINT8:
                return sizeof(uint8_t);
            case AIDB_DATA_TYPE_INT16:
                return sizeof(int16_t);
            case AIDB_DATA_TYPE_INT8:
                return sizeof(int8_t);
            case AIDB_DATA_TYPE_DOUBLE:
                return sizeof(double );
            case AIDB_DATA_TYPE_UINT16:
                return sizeof(uint16_t);
            case AIDB_DATA_TYPE_UINT32:
                return sizeof(uint32_t);
            case AIDB_DATA_TYPE_UINT64:
                return sizeof(uint64);
            case AIDB_DATA_TYPE_INT64:
                return sizeof(int64);
            default:
                return AIDB_DATA_TYPE_UNDEFINED;
        }

    }

#define CASE_TYPE(X)                             \
  case AIDB_DATA_TYPE_##X: \
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_##X;


    ONNXTensorElementDataType AiDB2OrtDataType(char const* str){
        switch (dataTypeFromStr(str)) {
            CASE_TYPE(FLOAT)
            CASE_TYPE(UINT8)
            CASE_TYPE(INT8)
            CASE_TYPE(UINT16)
            CASE_TYPE(INT16)
            CASE_TYPE(INT32)
            CASE_TYPE(INT64)
            CASE_TYPE(STRING)
            CASE_TYPE(BOOL)
            CASE_TYPE(FLOAT16)
            CASE_TYPE(DOUBLE)
            CASE_TYPE(UINT32)
            CASE_TYPE(UINT64)
            default:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        }
    }


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
                                                               std::accumulate(
                                                                       input_dim.begin(),
                                                                       input_dim.end(),
                                                                       1,
                                                                       std::multiplies<int>()
                                                               ) * sizeofA(dataTypeFromStr(_input_types[_input_node.first].c_str())),
                                                               input_dim.data(),
                                                               input_dim.size(), AiDB2OrtDataType(_input_types[_input_node.first].c_str()));
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
            }
            ort_output.GetTensorData<float>();
            outputs[index].resize(output_len);
            ::memcpy(outputs[index].data(), ort_output.GetTensorData<float>(), sizeof(float)*output_len);

        }
    }

    StatusCode ONNXEngine::init(const Parameter &param){
        _model_name = param._model_name;
        _backend_name = param._backend_name;
        _input_node_name.assign(param._input_node_name.begin(), param._input_node_name.end());
        _output_node_name.assign(param._output_node_name.begin(), param._output_node_name.end());

        _input_nodes = param._input_nodes;
        _input_types = param._input_types;

        _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, _model_name.c_str());

        // cuda
        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
        _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        _session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
#ifdef SD_ON
        if(!param._lora_path.empty()){
            std::vector<std::string> init_names;
            std::vector<Ort::Value> initializer_data;
            std::vector<onnx::TensorProto> tensors;
            auto allocator = Ort::AllocatorWithDefaultOptions();

            for(const auto& lora_path: param._lora_path){
                std::cout << lora_path << std::endl;
                std::ifstream fin(lora_path, std::ios::in | std::ios::binary);
                onnx::ModelProto onnx_model;
                onnx_model.ParseFromIstream(&fin);

                auto graph = onnx_model.graph();
                const auto& initializer = graph.initializer();

                for(auto& tensor: initializer){
                    init_names.push_back(tensor.name());
                    tensors.emplace_back(tensor);
                }

                fin.close();

            }

            for(const auto& tensor: tensors){
                std::vector<int64_t> shape(tensor.dims_size(), 0);
                for(int i = 0; i < tensor.dims_size(); i++){
                    shape[i] = tensor.dims(i);
                }

                Ort::Value input_tensor = Ort::Value::CreateTensor(allocator.GetInfo(),
                                                                   (void *)(tensor.raw_data().c_str()),
                                                                   tensor.raw_data().length(),
                                                                   shape.data(),
                                                                   shape.size(),
                                                                   ONNXTensorElementDataType(tensor.data_type()));

                initializer_data.push_back(std::move(input_tensor));
            }

            std::cout << init_names.size() << initializer_data.size() << std::endl;
            _session_options.AddExternalInitializers(init_names, initializer_data);
            _session = std::make_shared<Ort::Session>(_env, param._model_path.c_str(), _session_options);
            std::cout << "##" << param._model_path << "\n";
        } else{
            _session = std::make_shared<Ort::Session>(_env, param._model_path.c_str(), _session_options);
            std::cout << "@@" << param._model_path << "\n";
        }

#else
        _session = std::make_shared<Ort::Session>(_env, param._model_path.c_str(), _session_options);
#endif

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

    void ONNXEngine::forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape,
                             std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

        assert(_input_nodes.size() == input.size() && input_shape.size() == input.size());

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> ort_inputs;

        std::vector<char*> input_node_names;
        for(int i = 0; i < _input_node_name.size(); i++){
//        for (auto & _input_node : _input_nodes) {
            auto _input_node = _input_nodes[_input_node_name[i]];
            std::vector<int64_t> input_dim(_input_node.begin(), _input_node.end());
            for(int d = 0; d < input_dim.size(); d++){
                if(input_dim[d] < 0){
                    input_dim[d] = input_shape[i][d];
                }
            }

            Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info,
                                                               (void *) input[i],
                                                               std::accumulate(
                                                                       input_shape[i].begin(),
                                                                       input_shape[i].end(),
                                                                       1,
                                                                       std::multiplies<int>()
                                                               ) * sizeofA(dataTypeFromStr(_input_types[_input_node_name[i]].c_str())),
                                                               input_dim.data(),
                                                               input_dim.size(),
                                                               AiDB2OrtDataType(_input_types[_input_node_name[i]].c_str()));

            ort_inputs.push_back(std::move(input_tensor));
            input_node_names.push_back(const_cast<char*>(_input_node_name[i].c_str()));
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
            }

            ort_output.GetTensorData<float>();
            outputs[index].resize(output_len);
            ::memcpy(outputs[index].data(), ort_output.GetTensorData<float>(), sizeof(float)*output_len);

        }

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