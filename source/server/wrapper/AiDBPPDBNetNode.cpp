//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBPPDBNetNode.hpp"
#include <AIDBData.h>

#include "../reflect.hpp"

namespace AiDBServer {
    AiDB_REGISTER(AiDBPPDBNetNode)

    int AiDBPPDBNetNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));
        cv::Mat blob = *this->_ins << input;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                            outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::OcrMeta>> ocr_results;

        _post_process.dbnet_post_process(outputs[0], outputs_shape[0], ocr_results, this->_ins->scale_h(),
                                         this->_ins->scale_w(), input);

        result["ocr"] = {};

        for (auto &ocr_result: ocr_results) {
            std::vector<std::vector<int>> bbox;
            for (const auto &b: ocr_result->box) {
                bbox.push_back({b._x, b._y});
            }

            result["ocr"].push_back({{"box",  bbox},
                                     {"conf", ocr_result->conf}});

        }


        return 0;
    }
}