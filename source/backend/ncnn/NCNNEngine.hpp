#ifndef NCNN_MODEL_HPP
#define NCNN_MODEL_HPP

#include <memory>
#include "core/Engine.hpp"
#include "ncnn/net.h"

namespace AIDB{

    class NCNNEngine: public Engine{
    public:
        NCNNEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) override;
        ~NCNNEngine() override;
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
        void forward(std::vector<const void*> input, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
//        void forward(const float *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;

    private:

    private:
        std::shared_ptr<ncnn::Net> _net;
        ncnn::Option _opt;
//        ncnn::UnlockedPoolAllocator _ncnn_blob_pool_allocator;
//        ncnn::PoolAllocator _ncnn_workspace_pool_allocator;

    };

} //AIDB



#endif
