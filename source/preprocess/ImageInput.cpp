//
// Created by TalkUHulk on 2023/2/5.
//

#include "preprocess/ImageInput.hpp"

namespace AIDB {

    ImageInput::ImageInput(const YAML::Node &input_mode) {
#ifndef ENABLE_NCNN_WASM
        assert(input_mode["shape"].IsDefined());
        assert(input_mode["inputformat"].IsDefined());
        assert(input_mode["imageformat"].IsDefined());
#endif
        _shape.width = input_mode["shape"]["width"].IsDefined()? input_mode["shape"]["width"].as<int>(): 0;
        _shape.height = input_mode["shape"]["height"].IsDefined()? input_mode["shape"]["height"].as<int>(): 0;
        _shape.channel = input_mode["shape"]["channel"].IsDefined()? input_mode["shape"]["channel"].as<int>(): 0;

#ifdef ENABLE_NCNN_WASM
        if("RGB" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = RGB;
        else if("BGR" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = BGR;
        else if("GRAY" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = GRAY;
        else if("RGBA" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = RGBA;
        else if("BGRA" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = BGRA;
        else
            printf("Not Implement\n");

#endif

        if("RGB" == input_mode["imageformat"].as<std::string>())
            _image_format = RGB;
        else if("BGR" == input_mode["imageformat"].as<std::string>())
            _image_format = BGR;
        else if("GRAY" == input_mode["imageformat"].as<std::string>())
            _image_format = GRAY;
        else
            printf("Not Implement\n");

        if("NCHW" == input_mode["inputformat"].as<std::string>())
            _input_format = NCHW;
        else if("NHWC" == input_mode["inputformat"].as<std::string>())
            _input_format = NHWC;
        else
            printf("Not Implement\n");


        if(input_mode["mean"].IsDefined()){
            _mean.clear();
            _mean.resize(input_mode["mean"].size());
            for(int i=0;i < input_mode["mean"].size(); i++){
                _mean[i] = input_mode["mean"][i].as<float>();
            }
        }

        if(input_mode["var"].IsDefined()){
            _var.clear();
            _var.resize(input_mode["var"].size());
            for(int i=0;i < input_mode["var"].size(); i++){
                _var[i] = input_mode["var"][i].as<float>();
            }
        }

        if(input_mode["border_constant"].IsDefined()){
            _border_constant.clear();
            _border_constant.resize(input_mode["border_constant"].size());
            for(int i=0;i < input_mode["border_constant"].size(); i++){
                _border_constant[i] = input_mode["border_constant"][i].as<float>();
            }
        }

        if(input_mode["keep_ratio"].IsDefined())
            _keep_ratio = input_mode["keep_ratio"].as<bool>();

        if(input_mode["limit_side_len"].IsDefined())
            _limit_side_len = input_mode["limit_side_len"].as<int>();
        
    }

    ImageInput::~ImageInput() {

    }

#ifndef ENABLE_NCNN_WASM
    void ImageInput::forward(const cv::Mat &image, cv::Mat &blob) {

        assert(!image.empty());
        image.copyTo(_src_image);
        cv::Mat processed_image;
        Resize(image, processed_image);
//        cv::imwrite("b.png", processed_image);
//        cv::imshow("processed_image", processed_image);
//        cv::waitKey();
        cvtColor(processed_image, processed_image);
        Normalize(processed_image);
        int shape[] =  {_shape.channel, _shape.height, _shape.width};
        blob = cv::Mat(3, shape,  CV_32F);
        Permute(processed_image, blob);

    }
#else
    void ImageInput::forward(const cv::Mat &image, ncnn::Mat &blob){

//        ncnn::Mat blob = ncnn::Mat::from_pixels_resize(rgba.data, ncnn::Mat::PIXEL_RGBA2RGB, rgba.cols, rgba.rows, 640, 640);
//        const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
//        const float mean_vals[3] = {127.5, 127.5, 127.5};
//        blob.substract_mean_normalize(mean_vals, norm_vals);

        Resize(image, blob);


//        blob = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGBA2RGB, image.cols, image.rows, 640, 640);
        std::vector<float> norm_vals(_var);
        std::for_each(norm_vals.begin(), norm_vals.end(), [=](float &item){item = 1 / item;});
        blob.substract_mean_normalize(_mean.data(), norm_vals.data());
    }
#endif

#ifndef ENABLE_NCNN_WASM
    void ImageInput::forward(const std::string &image_path, cv::Mat &blob) {
        _src_image = cv::imread(image_path);
        assert(!_src_image.empty());
        cv::Mat processed_image;
        Resize(_src_image, processed_image);
//        cv::imshow("processed_image", processed_image);
//        cv::waitKey();
        cvtColor(processed_image, processed_image);

        Normalize(processed_image);
        int shape[] =  {_shape.channel, _shape.height, _shape.width};
        blob = cv::Mat(3, shape,  CV_32F);
        Permute(processed_image, blob);


    }


    void ImageInput::Normalize(cv::Mat &image) {
        image.convertTo(image, CV_32FC3);
        std::vector<cv::Mat> bgr_channels(image.channels());
        cv::split(image, bgr_channels);
        for (auto i = 0; i < bgr_channels.size(); i++) {
            bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 / _var[i],
                                      (0.0 - _mean[i]) / _var[i]);
        }
        cv::merge(bgr_channels, image);
    }

    void ImageInput::Permute(const cv::Mat &image, cv::Mat &blob) {

        if(NCHW == _input_format){
            cv::dnn::blobFromImage(image, blob);
        } else if(NHWC == _input_format){
            image.convertTo(blob, CV_32F);
        }
    }

    void ImageInput::Resize(const cv::Mat &image, cv::Mat &resized) {
        assert(!image.empty());
        int resize_w = _shape.width;
        int resize_h = _shape.height;

        if(resize_w == -1 && resize_h == -1){
            if(_limit_side_len > 0){
                float tmp_ratio = 1.f;
                int max_wh = std::max(image.cols, image.rows);
                if (max_wh > _limit_side_len) {
                    if (image.rows > image.cols) {
                        tmp_ratio = float(_limit_side_len) / float(image.rows);
                    } else {
                        tmp_ratio = float(_limit_side_len) / float(image.cols);
                    }
                }
                resize_h = int(float(image.rows) * tmp_ratio);
                resize_w = int(float(image.cols) * tmp_ratio);

                resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
                resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

            } else{
                resize_h = std::max(int(round(float(image.rows) / 32) * 32), 32);
                resize_w = std::max(int(round(float(image.cols) / 32) * 32), 32);
            }

            set_height(resize_h);
            set_width(resize_w);

        } else if(resize_h!= -1 && resize_w == -1){
            resize_w = resize_h / image.rows * image.cols;
            resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
            set_width(resize_w);

        } else if(resize_h == -1){
            resize_h = resize_w / image.cols * image.rows;
            resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
            set_height(resize_h);
        }


        if(_keep_ratio){
            float src_ratio = float(image.cols) / float(image.rows);
            float target_ratio = float(resize_w) / float(resize_h);

            int n_w, n_h, pad_w = 0, pad_h = 0;

            if(src_ratio > target_ratio){
                n_w = resize_w;

                n_h = floor(float(n_w) / float(image.cols) * float(image.rows) + 0.5f);
                pad_h = resize_h - n_h;
                _scale_h = _scale_w = float(n_w) / float(image.cols);
            } else if(src_ratio < target_ratio){
                n_h = resize_h;
                n_w = floor(float(n_h) / float(image.rows) * float(image.cols) + 0.5f);
                pad_w = resize_w - n_w;
                _scale_h = _scale_w = float(n_h) / float(image.rows);
            } else{
                n_w = resize_w;
                n_h = resize_h;
                _scale_h = _scale_w = float(n_w) / float(image.cols);
            }

            cv::resize(image, resized, cv::Size(n_w, n_h));
            cv::Scalar sc;
            if(_border_constant.size() == 1)
                sc = cv::Scalar::all(_border_constant[0]);
            else if(_border_constant.size() == 3)
                sc = cv::Scalar(_border_constant[0], _border_constant[1], _border_constant[2]);
            else if(_border_constant.size() == 4)
                sc = cv::Scalar(_border_constant[0], _border_constant[1], _border_constant[2], _border_constant[3]);

            cv::copyMakeBorder(resized, resized, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, sc);

        } else{
            cv::resize(image, resized, cv::Size(resize_w, resize_h));
            _scale_h = float(resize_h) / float(image.rows);
            _scale_w = float(resize_w) / float(image.cols);
        }
    }

    void ImageInput::cvtColor(const cv::Mat &image, cv::Mat &converted) {
        if(RGB == _image_format)
            cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
        else if(GRAY == _image_format)
            cv::cvtColor(image, converted, cv::COLOR_BGR2GRAY);
        else if(BGR == _image_format)
            converted = image;
        else
            printf("Not implement\n");
    }

#else
//    void ImageInput::Resize(const cv::Mat &image, ncnn::Mat &resized) {
//
//        int width = image.cols;
//        int height = image.rows;
//
//        int resize_w = _shape.width;
//        int resize_h = _shape.height;
//
//        if(resize_w == -1 && resize_h == -1){
//            if(_limit_side_len > 0){
//                float tmp_ratio = 1.f;
//                int max_wh = std::max(image.cols, image.rows);
//                if (max_wh > _limit_side_len) {
//                    if (image.rows > image.cols) {
//                        tmp_ratio = float(_limit_side_len) / float(image.rows);
//                    } else {
//                        tmp_ratio = float(_limit_side_len) / float(image.cols);
//                    }
//                }
//                resize_h = int(float(image.rows) * tmp_ratio);
//                resize_w = int(float(image.cols) * tmp_ratio);
//
//                resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
//                resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
//
//            } else{
//                resize_h = std::max(int(round(float(image.rows) / 32) * 32), 32);
//                resize_w = std::max(int(round(float(image.cols) / 32) * 32), 32);
//            }
//
//            set_height(resize_h);
//            set_width(resize_w);
//
//        } else if(resize_h!= -1 && resize_w == -1){
//            resize_w = resize_h / image.rows * image.cols;
//            resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
//            set_width(resize_w);
//
//        } else if(resize_h == -1){
//            resize_h = resize_w / image.cols * image.rows;
//            resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
//            set_height(resize_h);
//        }
//
//        int type =  ncnn::Mat::PIXEL_RGB;
//
//        if(RGBA == _origin_image_format && RGB == _image_format){
//            type = ncnn::Mat::PIXEL_RGBA2RGB;
//        } else if(RGBA == _origin_image_format && BGR == _image_format)
//            type = ncnn::Mat::PIXEL_RGBA2BGR;
//        else if(RGBA == _origin_image_format && GRAY == _image_format)
//            type = ncnn::Mat::PIXEL_RGBA2GRAY;
//        else if(RGB == _origin_image_format && BGR == _image_format)
//            type = ncnn::Mat::PIXEL_RGB2BGR;
//        else if(RGB == _origin_image_format && GRAY == _image_format)
//            type = ncnn::Mat::PIXEL_RGB2GRAY;
//        else if(BGR == _origin_image_format && RGB == _image_format)
//            type = ncnn::Mat::PIXEL_BGR2RGB;
//        else if(BGR == _origin_image_format && GRAY == _image_format)
//            type = ncnn::Mat::PIXEL_BGR2GRAY;
//
//
//        if(_keep_ratio){
//            float src_ratio = float(image.cols) / float(image.rows);
//            float target_ratio = float(resize_w) / float(resize_h);
//
//            int n_w, n_h, pad_w = 0, pad_h = 0;
//
//            if(src_ratio > target_ratio){
//                n_w = resize_w;
//
//                n_h = floor(float(n_w) / float(image.cols) * float(image.rows) + 0.5f);
//                pad_h = resize_h - n_h;
//                _scale_h = _scale_w = float(n_w) / float(image.cols);
//            } else if(src_ratio < target_ratio){
//                n_h = resize_h;
//                n_w = floor(float(n_h) / float(image.rows) * float(image.cols) + 0.5f);
//                pad_w = resize_w - n_w;
//                _scale_h = _scale_w = float(n_h) / float(image.rows);
//            } else{
//                n_w = resize_w;
//                n_h = resize_h;
//                _scale_h = _scale_w = float(n_w) / float(image.cols);
//            }
//
//
//            ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, type, width, height, n_w, n_h);
////            !!! type:1 n_w:640 n_h:480 width: 640 height:480 pad_h:160 pad_w:0
//            ncnn::copy_make_border(in, resized, 0, pad_h, 0, pad_w, ncnn::BORDER_CONSTANT, 114.f);
//
//
//        } else{
//            resized = ncnn::Mat::from_pixels_resize(image.data, type, width, height, resize_w, resize_h);
//            _scale_h = float(resize_h) / float(image.rows);
//            _scale_w = float(resize_w) / float(image.cols);
//        }
//
//    }

    void ImageInput::Resize(const cv::Mat &image, ncnn::Mat &resized) {

        auto width = has_roi()? _roi.width: image.cols;
        auto height = has_roi()? _roi.height: image.rows;

        int resize_w = _shape.width;
        int resize_h = _shape.height;

        if(resize_w == -1 && resize_h == -1){
            if(_limit_side_len > 0){
                float tmp_ratio = 1.f;
                int max_wh = std::max(width, height);
                if (max_wh > _limit_side_len) {
                    if (height > width) {
                        tmp_ratio = float(_limit_side_len) / float(height);
                    } else {
                        tmp_ratio = float(_limit_side_len) / float(width);
                    }
                }
                resize_h = int(float(height) * tmp_ratio);
                resize_w = int(float(width) * tmp_ratio);

                resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
                resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

            } else{
                resize_h = std::max(int(round(float(height) / 32) * 32), 32);
                resize_w = std::max(int(round(float(width) / 32) * 32), 32);
            }

            set_height(resize_h);
            set_width(resize_w);

        } else if(resize_h!= -1 && resize_w == -1){
            resize_w = resize_h / height * width;
            resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
            set_width(resize_w);

        } else if(resize_h == -1){
            resize_h = resize_w / width * height;
            resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
            set_height(resize_h);
        }

        int type =  ncnn::Mat::PIXEL_RGB;

        if(RGBA == _origin_image_format && RGB == _image_format){
            type = ncnn::Mat::PIXEL_RGBA2RGB;
        } else if(RGBA == _origin_image_format && BGR == _image_format)
            type = ncnn::Mat::PIXEL_RGBA2BGR;
        else if(RGBA == _origin_image_format && GRAY == _image_format)
            type = ncnn::Mat::PIXEL_RGBA2GRAY;
        else if(RGB == _origin_image_format && BGR == _image_format)
            type = ncnn::Mat::PIXEL_RGB2BGR;
        else if(RGB == _origin_image_format && GRAY == _image_format)
            type = ncnn::Mat::PIXEL_RGB2GRAY;
        else if(BGR == _origin_image_format && RGB == _image_format)
            type = ncnn::Mat::PIXEL_BGR2RGB;
        else if(BGR == _origin_image_format && GRAY == _image_format)
            type = ncnn::Mat::PIXEL_BGR2GRAY;


        if(_keep_ratio){
            float src_ratio = float(width) / float(height);
            float target_ratio = float(resize_w) / float(resize_h);

            int n_w, n_h, pad_w = 0, pad_h = 0;

            if(src_ratio > target_ratio){
                n_w = resize_w;

                n_h = floor(float(n_w) / float(width) * float(height) + 0.5f);
                pad_h = resize_h - n_h;
                _scale_h = _scale_w = float(n_w) / float(width);
            } else if(src_ratio < target_ratio){
                n_h = resize_h;
                n_w = floor(float(n_h) / float(height) * float(width) + 0.5f);
                pad_w = resize_w - n_w;
                _scale_h = _scale_w = float(n_h) / float(height);
            } else{
                n_w = resize_w;
                n_h = resize_h;
                _scale_h = _scale_w = float(n_w) / float(width);
            }


            ncnn::Mat in;
            if(has_roi()){
                in = ncnn::Mat::from_pixels_roi_resize(image.data, type, image.cols, image.rows, _roi.x, _roi.y, _roi.width, _roi.height, n_w, n_h);
            } else{
               in = ncnn::Mat::from_pixels_resize(image.data, type, width, height, n_w, n_h);
            }
            ncnn::copy_make_border(in, resized, 0, pad_h, 0, pad_w, ncnn::BORDER_CONSTANT, 114.f);


        } else{
            if(has_roi()){
                resized = ncnn::Mat::from_pixels_roi_resize(image.data, type, width, height, _roi.x, _roi.y, _roi.width, _roi.height, resize_w, resize_h);
                _scale_h = float(resize_h) / float(_roi.height);
                _scale_w = float(resize_w) / float(_roi.width);
            } else{
                resized = ncnn::Mat::from_pixels_resize(image.data, type, width, height, resize_w, resize_h);
                _scale_h = float(resize_h) / float(height);
                _scale_w = float(resize_w) / float(width);
            }


        }

    }
#endif

    ImageInput::ImageInput(const std::string &input_str) : AIDBInput(input_str) {

        auto input_mode = YAML::Load(input_str)["PreProcess"];
#ifndef ENABLE_NCNN_WASM
        assert(input_mode["shape"].IsDefined());
        assert(input_mode["inputformat"].IsDefined());
        assert(input_mode["imageformat"].IsDefined());
#endif

        _shape.width = input_mode["shape"]["width"].as<int>();
        _shape.height = input_mode["shape"]["height"].as<int>();
        _shape.channel = input_mode["shape"]["channel"].as<int>();

#ifdef ENABLE_NCNN_WASM
        if("RGB" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = RGB;
        else if("BGR" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = BGR;
        else if("GRAY" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = GRAY;
        else if("RGBA" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = RGBA;
        else if("BGRA" == input_mode["originimageformat"].as<std::string>())
            _origin_image_format = BGRA;
        else
            printf("Not Implement\n");
#endif

        if("RGB" == input_mode["imageformat"].as<std::string>())
            _image_format = RGB;
        else if("BGR" == input_mode["imageformat"].as<std::string>())
            _image_format = BGR;
        else if("GRAY" == input_mode["imageformat"].as<std::string>())
            _image_format = GRAY;
        else if("RGBA" == input_mode["imageformat"].as<std::string>())
            _image_format = RGBA;
        else if("BGRA" == input_mode["imageformat"].as<std::string>())
            _image_format = BGRA;
        else
            printf("Not Implement\n");

        if("NCHW" == input_mode["inputformat"].as<std::string>())
            _input_format = NCHW;
        else if("NHWC" == input_mode["inputformat"].as<std::string>())
            _input_format = NHWC;
        else
            printf("Not Implement\n");


        if(input_mode["mean"].IsDefined()){
            _mean.clear();
            _mean.resize(input_mode["mean"].size());
            for(int i=0;i < input_mode["mean"].size(); i++){
                _mean[i] = input_mode["mean"][i].as<float>();
            }
        }

        if(input_mode["var"].IsDefined()){
            _var.clear();
            _var.resize(input_mode["var"].size());
            for(int i=0;i < input_mode["var"].size(); i++){
                _var[i] = input_mode["var"][i].as<float>();
            }
        }

        if(input_mode["border_constant"].IsDefined()){
            _border_constant.clear();
            _border_constant.resize(input_mode["border_constant"].size());
            for(int i=0;i < input_mode["border_constant"].size(); i++){
                _border_constant[i] = input_mode["border_constant"][i].as<float>();
            }
        }

        if(input_mode["keep_ratio"].IsDefined())
            _keep_ratio = input_mode["keep_ratio"].as<bool>();

        if(input_mode["limit_side_len"].IsDefined())
            _limit_side_len = input_mode["limit_side_len"].as<int>();
    }


}