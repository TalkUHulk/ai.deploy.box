//
// Created by TalkUHulk on 2023/2/5.
//

#ifndef AIDEPLOYBOX_AIDBInput_HPP
#define AIDEPLOYBOX_AIDBInput_HPP

#include "AIDBDefine.h"
#include <string>
#include "core/Parameter.hpp"
#include "StatusCode.h"

#ifdef ENABLE_NCNN_WASM
#include <simpleocv.h>
#else
#include <opencv2/opencv.hpp>
#endif

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
        RGBA
    };

    enum InputFormat {
        NCHW = 0,
        NHWC
    };

    class AIDB_PUBLIC AIDBInput {
    public:
        AIDBInput() = default;
        explicit AIDBInput(const YAML::Node& input_mode){};
        explicit AIDBInput(const std::string& input_str){};
        virtual ~AIDBInput()= default;
#ifndef ENABLE_NCNN_WASM
        virtual void forward(const std::string &image_path, cv::Mat &blob) = 0;
        virtual void forward(const cv::Mat &image, cv::Mat &blob) = 0;
#else
        virtual void forward(const cv::Mat &image, ncnn::Mat &blob) = 0;
#endif


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

        void set_roi(cv::Rect roi){
            _roi = roi;
        }

        void set_roi(bool flag){
            _has_roi = flag;
        }

        bool has_roi() const{
            return _has_roi;
        }

    public:
        int _limit_side_len = 0; // 最长边限制
        float _scale_h = 1.0; // target / src
        float _scale_w = 1.0; // target / src
        cv::Mat _src_image;
    protected:
        bool _has_roi = false; // 是否单独处理roi
        cv::Rect _roi; // roi区域
        std::vector<float> _mean{.0f, .0f, .0f, .0f};
        std::vector<float> _var{1.0f, 1.0f, 1.0f, 1.0f};
        Shape _shape{};
        ImageFormat _origin_image_format;
        ImageFormat _image_format;
        InputFormat _input_format;
        bool _keep_ratio=true;
        std::vector<float> _border_constant{.0f, .0f, .0f, .0f};

    };
}

#endif //AIDEPLOYBOX_AIDBInput_HPP
