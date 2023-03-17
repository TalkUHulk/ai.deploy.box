//
// Created by TalkUHulk on 2023/2/5.
//

#ifndef AIDEPLOYBOX_IMAGEINPUT_HPP
#define AIDEPLOYBOX_IMAGEINPUT_HPP

#include "core/BaseInput.hpp"

//首先resize，然后swapRB，接着减mean，再乘以scalefactor
namespace AIDB{
    class ImageInput: public AIDBInput{
    public:
        explicit ImageInput(const YAML::Node& input_mode);
        ~ImageInput() override;
        void forward(const std::string &image_path, cv::Mat &blob) override;
        void forward(const cv::Mat &image, cv::Mat &blob) override;
    public:
        void Normalize(cv::Mat &image);
        void Permute(const cv::Mat &image, cv::Mat &blob);
        void Resize(const cv::Mat &image, cv::Mat &resized);
        void cvtColor(const cv::Mat &image, cv::Mat &converted);
    };
}



#endif //AIDEPLOYBOX_IMAGEINPUT_HPP
