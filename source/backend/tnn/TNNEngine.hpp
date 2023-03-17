#ifndef TNN_MODEL_HPP
#define TNN_MODEL_HPP

#include <memory>
#include "core/Engine.hpp"
#include "tnn/core/tnn.h"
#include "tnn/core/mat.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/core/macro.h"

namespace AIDB{

    class TNNEngine: public Engine{
    public:
        TNNEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const uint8_t *buffer_in, size_t buffer_size_in) override;
        ~TNNEngine() override;
        void forward(const float *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
        static std::string content_buffer_from(const char *proto_or_model_path);
    private:
        std::shared_ptr<tnn::TNN> _net;
        std::shared_ptr<tnn::Instance> _instance;
        tnn::DeviceType _input_device_type = tnn::DEVICE_X86; // only CPU, namely ARM or X86
        tnn::DeviceType _output_device_type = tnn::DEVICE_X86; // only CPU, namely ARM or X86
        tnn::DeviceType _network_device_type = tnn::DEVICE_X86; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM

    private:
        tnn::DimsVector get_input_shape(const std::string& name);
        tnn::DimsVector get_output_shape(const std::string& name);
        tnn::MatType get_output_mat_type(const std::string& name);
        tnn::DataFormat get_output_data_format(const std::string& name);
        tnn::MatType get_input_mat_type(const std::string& name);
        tnn::DataFormat get_input_data_format(const std::string& name);
        std::vector<std::string> get_input_names();
        std::vector<std::string> get_output_names();

    public:
        static tnn::DimsVector get_input_shape(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_);
        static tnn::DimsVector get_output_shape(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_);
        static std::vector<std::string> get_input_names(const std::shared_ptr<tnn::Instance> &instance_);
        static std::vector<std::string> get_output_names(const std::shared_ptr<tnn::Instance> &instance_);
        static tnn::MatType get_output_mat_type(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_);
        static tnn::DataFormat get_output_data_format(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_);
        static tnn::MatType get_input_mat_type(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_);
        static tnn::DataFormat get_input_data_format(const std::shared_ptr<tnn::Instance> &instance_, const std::string& name_);
    };

} //AIDB



#endif
