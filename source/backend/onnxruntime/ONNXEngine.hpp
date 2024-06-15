#ifndef _ONNX_RUNNER_H_
#define _ONNX_RUNNER_H_
#pragma once
#include <string>
#include <vector>
#include "core/Engine.hpp"
#include "OnnxRuntime/onnxruntime_cxx_api.h"

namespace AIDB{
    class ONNXEngine: public Engine{

    public:
        ONNXEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) override;

        ~ONNXEngine() override;
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
        void forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
    private:
        Ort::Env _env;
        Ort::SessionOptions _session_options;
        std::shared_ptr<Ort::Session> _session;
//        OrtMemoryInfo* _memory_info;
        std::map<std::string, std::vector<int>> _input_nodes;
        std::vector<std::string> _output_node_name;
        std::vector<std::string> _input_node_name;
        std::map<std::string, std::string> _input_types;
    };
} //AIDB

#endif
