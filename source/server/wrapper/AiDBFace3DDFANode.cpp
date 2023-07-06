//
// Created by TalkUHulk on 2023/7/3.
//

#include "AiDBFace3DDFANode.hpp"
#include <utility/Utility.h>
#include "../reflect.hpp"
#include "utility/td_obj.h"
//#include <fstream>
#include "cpp-base64/base64.h"

namespace AiDBServer {
    AiDB_REGISTER(AiDBFace3DDFANode)

    int AiDBFace3DDFANode::forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        if (result["face"].is_null() || result["face"].empty()) {
            spdlog::get(AIDB_DEBUG)->debug("No Face detected!");
            return -1;
        }

        cv::Mat input(frame_height, frame_width, CV_8UC3, const_cast<unsigned char *>(frame));

        auto output = input.clone();

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

            AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, input.cols, input.rows, 1.58, 0.14);
            cv::Mat roi(input, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(),
                                        face_meta_roi->height()));

            auto blob = *this->_ins << roi;
            outputs.clear();
            outputs_shape.clear();
            this->_ins->forward((float *) blob.data, this->_ins->width(), this->_ins->height(), this->_ins->channel(),
                                outputs, outputs_shape);
            std::vector<float> vertices, pose, sRt;

            AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

            result["face"][i]["pose"] = pose;

            if (outputs_shape[outputs_shape.size() - 1][1] > 204) {
                AIDB::Utility::TddfaUtility::tddfa_rasterize(output, vertices, spider_man_obj, 1, true);
            } else {
                for (int ii = 0; ii < vertices.size() / 3; ii++) {
                    cv::circle(output, cv::Point(vertices[3 * ii], vertices[3 * ii + 1]), 2, cv::Scalar(255, 255, 255),
                               -1);
                }
                AIDB::Utility::TddfaUtility::plot_pose_box(output, sRt, vertices, 68);
            }
        }

        std::vector<uchar> data_encode;

        int res = cv::imencode(".jpg", output, data_encode, {cv::IMWRITE_JPEG_QUALITY, 65});

//    std::ofstream fout("tddfa.jpg", std::ios::binary);
//    std::copy(data_encode.begin(), data_encode.end(), std::ostream_iterator<uchar>(fout, ""));
//    fout.close();

        std::string data_base64 = base64_encode(reinterpret_cast<const unsigned char *>(data_encode.data()),
                                                data_encode.size());
        result["tddfa"] = data_base64;

        return 0;
    }
}