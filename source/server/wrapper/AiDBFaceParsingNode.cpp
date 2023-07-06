//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBFaceParsingNode.hpp"
#include <utility/Utility.h>
#include "../reflect.hpp"
#include "utility/face_align.h"
#include "cpp-base64/base64.h"

namespace AiDBServer {
    AiDB_REGISTER(AiDBFaceParsingNode)

    int AiDBFaceParsingNode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {

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
            for (const auto &ldm: result["face"][i]["landmark"]) {
                face_meta->kps.push_back(ldm[0]);
                face_meta->kps.push_back(ldm[1]);
            }

            cv::Mat align;
            AIDB::faceAlign(input, align, face_meta, this->_ins->width(), "ffhq");

            auto blob = *this->_ins << align;

            cv::Mat src_image;
            *this->_ins >> src_image;

            outputs.clear();
            outputs_shape.clear();

            this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                                outputs, outputs_shape);

            cv::Mat output = src_image.clone();

            AIDB::Utility::bisenet_post_process(src_image, output, outputs[0], outputs_shape[0]);

            std::vector<uchar> data_encode;

            int res = imencode(".jpg", output, data_encode, {cv::IMWRITE_JPEG_QUALITY, 65});

            std::string data_base64 = base64_encode(reinterpret_cast<const unsigned char *>(data_encode.data()),
                                                    data_encode.size());

            result["face"][i]["parsing"] = data_base64;

        }

        return 0;
    }

}