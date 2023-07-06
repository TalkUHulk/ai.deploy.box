//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBFaceDetectNode.hpp"
#include <AIDBData.h>
#include <utility/Utility.h>
#include "../reflect.hpp"

namespace AiDBServer {
    AiDB_REGISTER(AiDBFaceDetectNode)

    int AiDBFaceDetectNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));
        cv::Mat blob = *this->_ins << input;

        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                            outputs, outputs_shape);

        std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;

        assert(face_detect_input->scale_h() == face_detect_input->scale_w());
        AIDB::Utility::scrfd_post_process(outputs,
                                          face_metas,
                                          this->_ins->width(),
                                          this->_ins->height(),
                                          this->_ins->scale_h());

        result["face"] = {};

        for (auto &face_meta: face_metas) {
            std::vector<float> bbox{face_meta->x1,
                                    face_meta->y1,
                                    face_meta->x2,
                                    face_meta->y2};

            std::vector<std::vector<float>> kps;
            for (int n = 0; n < face_meta->kps.size() / 2; n++) {
                kps.push_back({face_meta->kps[2 * n], face_meta->kps[2 * n + 1]});
            }
            result["face"].push_back({
                                             {"bbox",     bbox},
                                             {"landmark", kps},
                                             {"conf",     face_meta->score}});

        }
        return 0;
    }
}