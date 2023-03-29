#ifndef MNN_MODEL_HPP
#define MNN_MODEL_HPP

#include <memory>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include "core/Engine.hpp"



namespace MNN {
    class Session;
}

namespace AIDB{

    class MNNEngine: public Engine{
    public:
        MNNEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) override;
        ~MNNEngine() override;
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
    private:
        void reshape_input(const std::vector<int>&);
        std::shared_ptr<MNN::Tensor> get_output_by_name(const char *name);
        MNN::Tensor* get_input_tensor(const char *node_name);
        MNN::Tensor* get_input_tensor();

    private:
        std::shared_ptr<MNN::Interpreter> _mnn_net;
        MNN::ScheduleConfig _net_cfg;
        MNN::Session *_mnn_session;


    };

} //AIDB



#endif
