//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBMobileVitNode.hpp"
#include <AIDBData.h>
#include <utility/Utility.h>
#include "../reflect.hpp"

namespace AiDBServer {
    AiDB_REGISTER(AiDBMobileVitNode)

    int AiDBMobileVitNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));
        cv::Mat blob = *this->_ins << input;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                            outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::ClsMeta>> tmp;
        tmp.resize(outputs_shape[0][1]);
        for (int i = 0; i < outputs_shape[0][1]; i++) {
            std::shared_ptr<AIDB::ClsMeta> meta = std::make_shared<AIDB::ClsMeta>();
            meta->conf = outputs[0][i];
            meta->label = i;
            tmp[i] = meta;
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](const std::shared_ptr<AIDB::ClsMeta> &a, const std::shared_ptr<AIDB::ClsMeta> &b) {
                      return a->conf > b->conf;
                  });


        result["cls"] = {};
        for (int i = 0; i < _topK; i++) {

            result["cls"].push_back({
                                            {"label", tmp[i]->label},
                                            {"conf",  tmp[i]->conf}});

        }
        return 0;

    }
}
