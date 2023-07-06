//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBPPCLsNode.hpp"
#include <AIDBData.h>
#include "utility/log.h"
#include "../reflect.hpp"

namespace AiDBServer {
    AiDB_REGISTER(AiDBPPCLsNode)

    int AiDBPPCLsNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        if (result["ocr"].is_null() || result["ocr"].empty()) {
            spdlog::get(AIDB_DEBUG)->debug("No Ocr detected!");
            return -1;
        }

        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));

        for (int i = 0; i < result["ocr"].size(); i++) {

            std::vector<std::vector<float>> outputs;
            std::vector<std::vector<int>> outputs_shape;
            std::shared_ptr<AIDB::OcrMeta> ocr_result = std::make_shared<AIDB::OcrMeta>();
            for (auto &j : result["ocr"][i]["box"]) {
                ocr_result->box.emplace_back(j[0], j[1]);
            }
            ocr_result->conf = result["ocr"][i]["conf"];

            cv::Mat crop_img;
            AIDB::Utility::PPOCR::GetRotateCropImage(input, crop_img, ocr_result);

            cv::Mat cls_blob = *this->_ins << crop_img;
            this->_ins->forward((float *) cls_blob.data, this->_ins->width(), this->_ins->height(),
                                this->_ins->channel(), outputs, outputs_shape);

            result["ocr"][i]["conf_rotate"] = outputs[0][1];

        }


        return 0;
    }
}