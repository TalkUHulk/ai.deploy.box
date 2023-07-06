//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBYoloV8Node.hpp"
#include <AIDBData.h>
#include <utility/Utility.h>
#include "../reflect.hpp"

namespace AiDBServer {
    AiDB_REGISTER(AiDBYoloV8Node)

    int AiDBYoloV8Node::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));
        cv::Mat blob = *this->_ins << input;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                            outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::ObjectMeta>> object;

        AIDB::Utility::yolov8_post_process(outputs[0], outputs_shape[0], object, 0.45, 0.35, this->_ins->scale_h());

        result["object"] = {};

        for (const auto &obj : object) {

            std::vector<float> bbox{obj->x1,
                                    obj->y1,
                                    obj->x2,
                                    obj->y2};

            result["object"].push_back({{"bbox",  bbox},
                                        {"label", obj->label},
                                        {"conf",  obj->score}});

        }
        return 0;
    }
}