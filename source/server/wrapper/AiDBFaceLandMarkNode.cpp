//
// Created by TalkUHulk on 2023/7/3.
//

#include <AIDBData.h>
#include <utility/Utility.h>
#include "AiDBFaceLandMarkNode.hpp"
#include "../reflect.hpp"

namespace AiDBServer {
    AiDB_REGISTER(AiDBFaceLandMarkNode)

    int AiDBFaceLandMarkNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {

        if (result["face"].is_null() || result["face"].empty()) {
            spdlog::get(AIDB_DEBUG)->debug("No Face detected!");
            return -1;
        }

        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));

        for (int i = 0; i < result["face"].size(); i++) {

            std::vector<std::vector<float>> outputs;
            std::vector<std::vector<int>> outputs_shape;
            std::shared_ptr<AIDB::FaceMeta> face_meta = std::make_shared<AIDB::FaceMeta>();

            face_meta->score = result["face"][i]["conf"];

            face_meta->x1 = result["face"][i]["bbox"][0];
            face_meta->y1 = result["face"][i]["bbox"][1];
            face_meta->x2 = result["face"][i]["bbox"][2];
            face_meta->y2 = result["face"][i]["bbox"][3];
            std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
            AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, input.cols, input.rows, 1.28, 0.14);
            cv::Mat roi(input, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(),
                                        face_meta_roi->height()));

            auto blob = *this->_ins << roi;
            this->_ins->forward((float *) blob.data,
                                this->_ins->width(),
                                this->_ins->height(),
                                this->_ins->channel(),
                                outputs,
                                outputs_shape);
            AIDB::Utility::pfpld_post_process(outputs, face_meta_roi, face_meta, 98);

            std::vector<std::vector<float>> kps;
            kps.reserve(face_meta->kps.size() / 2);
            for (int n = 0; n < face_meta->kps.size() / 2; n++) {
                kps.push_back({face_meta->kps[2 * n], face_meta->kps[2 * n + 1]});
            }
            result["face"][i]["landmark"] = kps;
        }

        return 0;
    }

}