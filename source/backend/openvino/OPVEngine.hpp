#ifndef OPV_MODEL_HPP
#define OPV_MODEL_HPP

#include <memory>
#include "core/Engine.hpp"
#include "openvino/openvino.hpp"


namespace AIDB{

    class OPVEngine: public Engine{
    public:
        OPVEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) override;
        ~OPVEngine() override;
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
        void forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
//        void preprocess(unsigned char *frame, int frame_width, int frame_height) override;
//        void postprecess() override;
    private:



    private:
        ov::Core _core;
        std::shared_ptr<ov::Model> _model;
        ov::CompiledModel _compiled_model;
        ov::InferRequest _infer_request;
        ov::element::Type _input_type = ov::element::f32; //暂时写死
        const ov::Layout _input_layout{"NCHW"}; //暂时写死
    };

} //AIDB



#endif
