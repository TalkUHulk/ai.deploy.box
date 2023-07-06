//
// Created by TalkUHulk on 2023/7/4.
//

#include "AiDBAnimeGanNode.hpp"
#include <AIDBData.h>
#include <utility/Utility.h>
#include "../reflect.hpp"
#include "cpp-base64/base64.h"

namespace AiDBServer {
    AiDB_REGISTER(AiDBAnimeGanNode)

    int AiDBAnimeGanNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));
        cv::Mat blob = *this->_ins << input;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                            outputs, outputs_shape);

        auto output = input.clone();

        AIDB::Utility::animated_gan_post_process(outputs[0], outputs_shape[0], output);

        std::vector<uchar> data_encode;

        int res = imencode(".jpg", output, data_encode, {cv::IMWRITE_JPEG_QUALITY, 65});

        std::string data_base64 = base64_encode(reinterpret_cast<const unsigned char *>(data_encode.data()),
                                                data_encode.size());

        result["anime"] = data_base64;

        return 0;
    }
}