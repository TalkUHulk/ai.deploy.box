//
// Created by TalkUHulk on 2023/2/5.
//

#ifndef AIDEPLOYBOX_AIDBInput_HPP
#define AIDEPLOYBOX_AIDBInput_HPP

#include "AIDBDefine.h"
#include <string>
#include "core/Parameter.hpp"
#include "StatusCode.h"
#include <opencv2/opencv.hpp>

namespace AIDB {

    struct Shape{
        int width;
        int height;
        int channel;
    };

    enum ImageFormat {
        RGB = 0,
        BGR,
        GRAY,
        BGRA,
    };

    enum InputFormat {
        NCHW = 0,
        NHWC
    };

    class AIDB_PUBLIC AIDBInput {
    public:
        AIDBInput() = default;
        explicit AIDBInput(const YAML::Node& input_mode){};
        virtual ~AIDBInput()= default;
        virtual void forward(const std::string &image_path, cv::Mat &blob) = 0;
        virtual void forward(const cv::Mat &image, cv::Mat &blob) = 0;

        int width() const{
            return _shape.width;
        }
        int height() const{
            return _shape.height;
        }
        int channel() const{
            return _shape.channel;
        }

        void set_width(int width){
            _shape.width = width;
        }
        void set_height(int height){
            _shape.height = height;
        }
        void set_channel(int channel){
            _shape.channel = channel;
        }

    public:
        int _limit_side_len = 0; // 最长边限制
        float _scale_h = 1.0; // target / src
        float _scale_w = 1.0; // target / src
        cv::Mat _src_image;
    protected:
        std::vector<float> _mean{.0f, .0f, .0f, .0f};
        std::vector<float> _var{1.0f, 1.0f, 1.0f, 1.0f};
        Shape _shape{};
        ImageFormat _image_format;
        InputFormat _input_format;
        bool _keep_ratio=true;
        std::vector<float> _border_constant{.0f, .0f, .0f, .0f};

    };
}

#endif //AIDEPLOYBOX_AIDBInput_HPP
