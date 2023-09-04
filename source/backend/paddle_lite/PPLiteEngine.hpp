#ifndef PPLITE_MODEL_HPP
#define PPLITE_MODEL_HPP

#include <memory>
#include "core/Engine.hpp"
#include "paddle_api.h"


namespace AIDB{

    class PPLiteEngine: public Engine{
    public:
        PPLiteEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) override;
        ~PPLiteEngine() override;
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
        void forward(std::vector<const void*> input, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
//        void preprocess(unsigned char *frame, int frame_width, int frame_height) override;
//        void postprecess() override;
    private:



    private:
        paddle::lite_api::MobileConfig _config;
        std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor;
    };

} //AIDB



#endif
