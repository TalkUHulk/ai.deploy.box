//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBMoveNetNode.hpp"
#include <AIDBData.h>
#include <utility/Utility.h>
#include "../reflect.hpp"

namespace AiDBServer {

    AiDB_REGISTER(AiDBMoveNetNode)

    int AiDBMoveNetNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));
        cv::Mat blob = *this->_ins << input;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                            outputs, outputs_shape);

        std::vector<std::vector<float>> decoded_keypoints;

        AIDB::Utility::movenet_post_process(input, outputs, outputs_shape, decoded_keypoints);

        std::vector<std::vector<float>> dkps;

        dkps.reserve(decoded_keypoints.size());

        for (auto &kps: decoded_keypoints) {
            dkps.push_back({kps[0], kps[1]});
        }

        result["key_points"] = dkps;

        return 0;
    }
}
