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
        StatusCode init(const Parameter&, const uint8_t *buffer_in, size_t buffer_size_in) override;
        ~ONNXEngine() override;
        void forward(const float *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;

    private:
        Ort::Env _env;
        Ort::SessionOptions _session_options;
        std::shared_ptr<Ort::Session> _session;
//        OrtMemoryInfo* _memory_info;
    };
} //AIDB

#endif
