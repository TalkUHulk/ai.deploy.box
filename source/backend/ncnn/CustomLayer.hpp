//
// Created by TalkUHulk on 2023/2/26.
//

#ifndef AIDEPLOYBOX_CUSTOMLAYER_HPP
#define AIDEPLOYBOX_CUSTOMLAYER_HPP
#include "ncnn/layer.h"
#include "ncnn/net.h"
#include <map>

// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};


DEFINE_LAYER_CREATOR(YoloV5Focus);

typedef ncnn::Layer* (*custom_layer_func)(void*);


static std::map<std::string, custom_layer_func> custom_layer_map{{"YoloV5Focus", YoloV5Focus_layer_creator}};


#endif //AIDEPLOYBOX_CUSTOMLAYER_HPP
