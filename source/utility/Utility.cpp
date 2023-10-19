//
// Created by TalkUHulk on 2023/2/5.
//

#include "Utility.h"
#include "AIDBDefine.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <utility>
#include "rasterize.h"
#include "triangles.hpp"
#include "center_weight.h"
#include "labels.h"
#include "clipper.h"
#include <chrono>
#include <thread>

//#if !defined(ENABLE_NCNN_WASM) && !defined(__ANDROID__)
//#include <opencv2/freetype.hpp>
//#endif
#ifdef OPENCV_HAS_FREETYPE
#include <opencv2/freetype.hpp>
#endif

namespace AIDB {
#define AIDB_PI 3.1415926
    template<class T>
    float Utility::Common::IOU(const std::shared_ptr<T> meta1, const std::shared_ptr<T> meta2) {
        float x1 = fmax(meta1->x1, meta2->x1);
        float y1 = fmax(meta1->y1, meta2->y1);
        float x2 = fmin(meta1->x2, meta2->x2);
        float y2 = fmin(meta1->y2, meta2->y2);
        float w = fmax(.0f, x2 - x1);
        float h = fmax(.0f, y2 - y1);
        float over_area = w * h;
        return over_area / (meta1->area() + meta2->area() - over_area);
    }

    template<class T>
    void Utility::Common::NMS(std::vector<std::shared_ptr<T>> &metas, std::vector<std::shared_ptr<T>> &keep_metas, float threshold) {
//        std::sort(metas.begin(), metas.end(),
//                  [](const std::shared_ptr<T> meta1, const std::shared_ptr<T> meta2) { return meta1->score > meta2->score; });
//        while (!metas.empty()) {
//            keep_metas.push_back(metas[0]);
//            int index = 1;
//            while (index < metas.size()) {
//                float iou_value = IOU<T>(metas[0], metas[index]);
//                if (iou_value > threshold) {
//                    metas.erase(metas.begin() + index);
//                } else {
//                    index++;
//                }
//
//            }
//            metas.erase(metas.begin());
//        }
        keep_metas.clear();
        std::vector<int> picked;
        const int n = metas.size();
        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) {
            areas[i] = metas[i]->area();
        }

        for (int i = 0; i < n; i++) {
            const auto &a = metas[i];
            int keep = 1;
            for (const auto j : picked) {
                const auto &b = metas[j];
                float iou = IOU<T>(a, b);
                if (iou > threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
        for(int i: picked)
            keep_metas.push_back(metas[i]);

    }

    void Utility::Common::parse_roi_from_bbox(const std::shared_ptr<FaceMeta> org_meta, std::shared_ptr<FaceMeta> dst_meta, int image_w, int image_h,
                                      float expand_ratio, float re_centralize_ratio) {

        float old_size = (org_meta->y2 - org_meta->y1 + org_meta->x2 - org_meta->x1) / 2;
        float cx = org_meta->x2 - (org_meta->x2 - org_meta->x1) / 2;
        float cy = org_meta->y2 - (org_meta->y2 - org_meta->y1) / 2 + old_size * re_centralize_ratio;
        int new_size = int(old_size * expand_ratio);

        dst_meta->score = org_meta->score;
        dst_meta->kps.assign(org_meta->kps.begin(), org_meta->kps.end());

        dst_meta->x1 = cx - new_size / 2;
        dst_meta->y1 = cy - new_size / 2;
        dst_meta->x2 = dst_meta->x1 + new_size;
        dst_meta->y2 = dst_meta->y1 + new_size;

        if (dst_meta->x1 < 0) {
            dst_meta->x2 -= dst_meta->x1;
            dst_meta->x1 = 0;
        }
        if (dst_meta->y1 < 0) {
            dst_meta->y2 -= dst_meta->y1;
            dst_meta->y1 = 0;
        }
        if (dst_meta->x2 >= image_w) dst_meta->x2 = image_w - 1;
        if (dst_meta->y2 >= image_h) dst_meta->y2 = image_h - 1;

    }

    void Utility::Common::draw_objects(const cv::Mat &src, cv::Mat &dst, const vector<std::shared_ptr<ObjectMeta>> &objects) {

//        src.copyTo(dst);
        dst = src.clone();
        for (const auto & object : objects){


            cv::rectangle(dst, cv::Point(object->x1, object->y1), cv::Point(object->x2, object->y2), cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", coco_labels[object->label], object->score * 100);
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = object->x1;
            int y = object->y1 - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > dst.cols)
                x = dst.cols - label_size.width;

            cv::rectangle(dst, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(dst, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }

    void Utility::scrfd_post_process(const std::vector<std::vector<float>> &outputs, std::vector<std::shared_ptr<FaceMeta>> &face_metas,
                                     int target_w, int target_h, float det_scale, float nms_thresh) {
        std::vector<int> feat_stride_fpn;
        int fmc;
        int num_anchors;
        bool use_kps = false;
        switch (outputs.size()) {
            case 6: {
                feat_stride_fpn.push_back(8);
                feat_stride_fpn.push_back(16);
                feat_stride_fpn.push_back(32);
                fmc = 3;
                num_anchors = 2;
            }
                break;
            case 9: {
                feat_stride_fpn.push_back(8);
                feat_stride_fpn.push_back(16);
                feat_stride_fpn.push_back(32);
                fmc = 3;
                num_anchors = 2;
                use_kps = true;
            }
                break;
            case 10: {
                feat_stride_fpn.push_back(8);
                feat_stride_fpn.push_back(16);
                feat_stride_fpn.push_back(32);
                feat_stride_fpn.push_back(64);
                feat_stride_fpn.push_back(128);
                fmc = 5;
                num_anchors = 1;
            }
                break;
            case 15: {
                feat_stride_fpn.push_back(8);
                feat_stride_fpn.push_back(16);
                feat_stride_fpn.push_back(32);
                feat_stride_fpn.push_back(64);
                feat_stride_fpn.push_back(128);
                fmc = 5;
                num_anchors = 1;
                use_kps = true;
            }
                break;
        }


        std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
        std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;


        std::vector<std::shared_ptr<FaceMeta>> _face_metas;
        for (int idx = 0; idx < feat_stride_fpn.size(); idx++) {
            auto stride = feat_stride_fpn[idx];
            std::vector<float> scores = outputs[idx];
            std::vector<float> bbox_preds = outputs[idx + fmc];
            for (auto &b: bbox_preds) {
                b *= float(stride);
            }
            std::vector<float> kps_preds;
            if (use_kps) {
                kps_preds = outputs[idx + fmc * 2];
                for (auto &b: kps_preds) {
                    b *= float(stride);
                }
            }

            int height = int(target_h / stride);
            int width = int(target_w / stride);

            float thresh = 0.5;
//            std::vector<int> pos_inds;
            std::vector<std::vector<int>> pos_inds3d;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int n = 0; n < num_anchors; n++) {
                        if (scores[y * width * num_anchors + x * num_anchors + n] >= thresh) {
                            pos_inds3d.push_back({y, x, n});
                        }
                    }
                }
            }
//            for (int pos = 0; pos < scores.size(); pos++) {
//                if (scores[pos] >= thresh) {
//                    pos_inds.push_back(pos);
//                }
//            }

//            if(pos_inds.empty())
//                continue;

            // bbox
            std::vector<std::vector<float>> bbox;
            std::vector<std::vector<float>> kps;


//            for (int y = 0; y < height; y++) {
//                for (int x = 0; x < width; x++) {
//                    int anchor_x = x * stride;
//                    int anchor_y = y * stride;
//                    for (int n = 0; n < num_anchors; n++) {
//                        int pos = (y * width * 2 + x * 2 + n) * 4;
//                        float x1 = anchor_x - bbox_preds[pos];
//                        float y1 = anchor_y - bbox_preds[pos + 1];
//                        float x2 = anchor_x + bbox_preds[pos + 2];
//                        float y2 = anchor_y + bbox_preds[pos + 3];
//                        std::vector<float> _box{x1, y1, x2, y2};
//                        bbox.push_back(_box);
//                        if (use_kps) {
//                            int kpos = (y * width * 2 + x * 2 + n) * 10;
//                            std::vector<float> _kps;
//                            for (int k = 0; k < 5; k++) {
//                                float kx = anchor_x + kps_preds[kpos + k * 2];
//                                float ky = anchor_y + kps_preds[kpos + k * 2 + 1];
//                                _kps.push_back(kx);
//                                _kps.push_back(ky);
//
//                            }
//                            kps.push_back(_kps);
//                        }
//                    }
//                }
//            }


            for (auto _pos: pos_inds3d) {
                auto y = _pos[0];
                auto x = _pos[1];
                auto n = _pos[2];
                int anchor_x = x * stride;
                int anchor_y = y * stride;

                int pos = y * width * num_anchors + x * num_anchors + n;
                int pos_bbox = (y * width * 2 + x * 2 + n) * 4;
                float x1 = anchor_x - bbox_preds[pos_bbox];
                float y1 = anchor_y - bbox_preds[pos_bbox + 1];
                float x2 = anchor_x + bbox_preds[pos_bbox + 2];
                float y2 = anchor_y + bbox_preds[pos_bbox + 3];

                std::shared_ptr<FaceMeta> face_meta = std::make_shared<FaceMeta>();

                face_meta->score = scores[pos];
                face_meta->x1 = x1 / det_scale;
                face_meta->y1 = y1 / det_scale;
                face_meta->x2 = x2 / det_scale;
                face_meta->y2 = y2 / det_scale;

                if (use_kps) {
                    int pos_kps = (y * width * 2 + x * 2 + n) * 10;
                    std::vector<float> _kps;
                    for (int k = 0; k < 5; k++) {
                        float kx = anchor_x + kps_preds[pos_kps + k * 2];
                        float ky = anchor_y + kps_preds[pos_kps + k * 2 + 1];
                        face_meta->kps.push_back(kx);
                        face_meta->kps.push_back(ky);

                    }
                    for_each(face_meta->kps.begin(), face_meta->kps.end(), [&](float &k) { k /= det_scale; });

                }

                _face_metas.push_back(face_meta);

            }

//            for (auto _pos: pos_inds) {
////                FaceMeta face_meta;
//                std::shared_ptr<FaceMeta> face_meta = std::make_shared<FaceMeta>();
//                face_meta->score = scores[_pos];
//                face_meta->x1 = bbox[_pos][0] / det_scale;
//                face_meta->y1 = bbox[_pos][1] / det_scale;
//                face_meta->x2 = bbox[_pos][2] / det_scale;
//                face_meta->y2 = bbox[_pos][3] / det_scale;
//                face_meta->kps.resize(kps[_pos].size());
//                face_meta->kps.assign(kps[_pos].begin(), kps[_pos].end());
//                for_each(face_meta->kps.begin(), face_meta->kps.end(), [&](float &k) { k /= det_scale; });
//                _face_metas.push_back(face_meta);
//
//            }
        }


        face_metas.clear();
        Common::NMS<FaceMeta>(_face_metas, face_metas, nms_thresh);

    }

    void Utility::pfpld_post_process(const std::vector<std::vector<float>> &outputs,
                                     const std::shared_ptr<FaceMeta> scale_face_metas, std::shared_ptr<FaceMeta> dst_face_metas, int pts_num) {

//        assert(2 == outputs.size());
        auto pose_pred = outputs[0];
        auto landmark_pred = outputs[1];
        for (auto &p: pose_pred) {
            p *= 180 / AI_PI;
        }

        for (int i = 0; i < pts_num; i++) {
            landmark_pred[i * 2] = landmark_pred[i * 2] * scale_face_metas->width() + scale_face_metas->x1;
            landmark_pred[i * 2 + 1] = landmark_pred[i * 2 + 1] * scale_face_metas->height() + scale_face_metas->y1;
        }
        dst_face_metas->pose.assign(pose_pred.begin(), pose_pred.begin() + 3);
        dst_face_metas->kps.clear();
        dst_face_metas->kps.resize(pts_num * 2);
        dst_face_metas->kps.assign(landmark_pred.begin(), landmark_pred.begin() + pts_num * 2);

    }

    void Utility::movenet_post_process(const cv::Mat &src_image,
                                       cv::Mat &result_image,
                                       const vector<std::vector<float>> &outputs,
                                       const vector<std::vector<int>> &outputs_shape) {
        std::vector<std::vector<float>> decoded_keypoints;

//        src_image.copyTo(result_image);
        if(result_image.data != src_image.data)
            result_image = src_image.clone();

        movenet_post_process(src_image, outputs, outputs_shape, decoded_keypoints);

        for(auto kps: decoded_keypoints){
            cv::circle(result_image, cv::Point(kps[0], kps[1]), 3, cv::Scalar(0, 255, 0), -1);
        }

        for(auto & index : AIDB::Utility::MoveNetUtility::line_map){
            cv::line(result_image, cv::Point(decoded_keypoints[index[0]][0], decoded_keypoints[index[0]][1]),
                     cv::Point(decoded_keypoints[index[1]][0], decoded_keypoints[index[1]][1]),
                     cv::Scalar(0, 255, 255), 2);
        }
    }

    void Utility::movenet_post_process(const cv::Mat &src_image,
                                       const vector<std::vector<float>> &outputs,
                                       const vector<std::vector<int>> &outputs_shape,
                                       std::vector<std::vector<float>>& decoded_keypoints) {


        AIDB::Utility::MoveNetUtility::MoveNetDecode(outputs, outputs_shape, decoded_keypoints);

        for_each(decoded_keypoints.begin(), decoded_keypoints.end(), [=](std::vector<float> &kps){ kps[0] *= src_image.cols; kps[1] *= src_image.rows; });

    }

    void Utility::TddfaUtility::similar_transform(const std::vector<float> &input, int bs, int pts3d_num,
                           std::vector<float> &output,
                           const std::shared_ptr<FaceMeta>& facemeta, int target){

        output.clear();
        output.resize(input.size());

        float scale_x = (facemeta->x2 - facemeta->x1) / target;
        float scale_y = (facemeta->y2 - facemeta->y1) / target;

        auto s = (scale_x + scale_y) / 2.0f;

        for(int n = 0; n < bs; n++){
            float min_z = std::numeric_limits<float>::max();
            for(int i = 0; i < pts3d_num; i++){
                output[n * pts3d_num * 3 + i] = (input[n * pts3d_num * 3 + i] - 1) * scale_x + facemeta->x1; // x
                output[n * pts3d_num * 3 + pts3d_num + i] = (target - input[n * pts3d_num * 3 + pts3d_num + i]) * scale_y + facemeta->y1; // y
                output[n * pts3d_num * 3 + 2 * pts3d_num + i] *= s; // z
                min_z = output[n * pts3d_num * 3 + 2 * pts3d_num + i] < min_z? output[n * pts3d_num * 3 + 2 * pts3d_num + i]: min_z;
            }

            for(int i = 0; i < pts3d_num; i++){
                output[n * pts3d_num * 3 + 2 * pts3d_num + i] -= min_z;
            }
        }


    }


    void Utility::TddfaUtility::matrix2angle(const std::vector<float> &sRt, std::vector<float> &pose){
        // compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
        // refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
        pose.clear();
        pose.resize(3);
        if(sRt[8] > 0.998){
            pose[2] = 0;
            pose[0] = AIDB_PI / 2;
            pose[1] = pose[2] + atan2(-sRt[1], -sRt[2]);
        } else if(sRt[8] < -0.998){
            pose[2] = 0;
            pose[0] = -AIDB_PI / 2;
            pose[1] = -pose[2] + atan2(-sRt[1], -sRt[2]);
        } else{
            pose[0] = asin(sRt[8]);
            pose[1] = atan2(sRt[9] / cos(pose[0]), sRt[10] / cos(pose[0]));
            pose[2] = atan2(sRt[4] / cos(pose[0]), sRt[0] / cos(pose[0]));
        }
    }


    void Utility::TddfaUtility::calc_pose(const std::vector<float> &input, std::vector<float> &sRt, std::vector<float> &pose) {

        sRt.resize(12);
        sRt.assign(input.begin(), input.end());
        float r1_norm = std::sqrt(std::pow(sRt[0], 2.0) + std::pow(sRt[1], 2.0) + std::pow(sRt[2], 2.0));
        float r2_norm = std::sqrt(std::pow(sRt[4], 2.0) + std::pow(sRt[5], 2.0) + std::pow(sRt[6], 2.0));
        // r1
        for(int i=0; i<3; i++){
            sRt[i] /= r1_norm;
        }
        // r2
        for(int i=4; i<7; i++){
            sRt[i] /= r2_norm;
        }
        // r3
        sRt[8] = sRt[1] * sRt[6] - sRt[2] * sRt[5];
        sRt[9] = sRt[2] * sRt[4] - sRt[0] * sRt[6];
        sRt[10] = sRt[0] * sRt[5] - sRt[1] * sRt[4];

        matrix2angle(sRt, pose);
        for(auto &p: pose){
            p *= (180 / AIDB_PI);
        }
    }

    void Utility::TddfaUtility::calc_hypotenuse_and_mean(const std::vector<float> &ver, float &mean_x, float &mean_y, float &llength, int kpts_num, int mean_index_end){
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        int step = ver.size() / kpts_num;
        mean_x = 0;
        mean_y = 0;
        for(int i = 0; i < kpts_num; i++){
            auto x = ver[i * step];
            auto y = ver[i * step + 1];
            min_x = fmin(min_x, x);
            min_y = fmin(min_y, y);
            max_x = fmax(max_x, x);
            max_y = fmax(max_y, y);
            if(i < mean_index_end){
                mean_x += x;
                mean_y += y;
            }
        }
        mean_x /= mean_index_end;
        mean_y /= mean_index_end;

        float radius = fmax(max_x - min_x, max_y - min_y) / 2.0f;

        llength = sqrt(8) * radius / 3;

    }

    Eigen::Matrix<float, 10, 4, Eigen::RowMajor> Utility::TddfaUtility::build_camera_box(float rear_size){

        Eigen::Matrix<float,10, 4, Eigen::RowMajor> point_3d_homo;
        float rear_depth = 0;
        int front_size = int(4.0f / 3.0f * rear_size);
        int front_depth = int(4.0f / 3.0f * rear_size);

        point_3d_homo << -rear_size, -rear_size, rear_depth, 1,
                -rear_size, rear_size, rear_depth, 1,
                rear_size, rear_size, rear_depth, 1,
                rear_size, -rear_size, rear_depth, 1,
                -rear_size, -rear_size, rear_depth, 1,
                -front_size, -front_size, front_depth, 1,
                -front_size, front_size, front_depth, 1,
                front_size, front_size, front_depth, 1,
                front_size, -front_size, front_depth, 1,
                -front_size, -front_size, front_depth, 1;
        return point_3d_homo;
    }
#ifndef ENABLE_NCNN_WASM
    void Utility::TddfaUtility::plot_pose_box(cv::Mat &img, const std::vector<float> &P, const std::vector<float> &ver, int kpts_num, cv::Scalar color, int line_width){

        float mean_ver_x;
        float mean_ver_y;
        float llength;
        calc_hypotenuse_and_mean(ver, mean_ver_x, mean_ver_y, llength, kpts_num);

        Eigen::Matrix<float, 10, 4, Eigen::RowMajor> mat_point_3d_homo = build_camera_box(llength);
        Eigen::Map<Eigen::Matrix<float,3, 4, Eigen::RowMajor>> matP(const_cast<float*>(P.data()));

        Eigen::Matrix<float, 10, 3, Eigen::RowMajor> point_3d = mat_point_3d_homo * matP.transpose();

        float mean_pts_x = (point_3d.coeff(0, 0) + point_3d.coeff(1, 0) + point_3d.coeff(2, 0) + point_3d.coeff(3, 0)) / 4;
        float mean_pts_y = -(point_3d.coeff(0, 1) + point_3d.coeff(1, 1) + point_3d.coeff(2, 1) + point_3d.coeff(3, 1)) / 4;

        cv::Point pts[1][10];
        for(int i = 0; i < 10; i++){
            pts[0][i] = cv::Point(point_3d.coeff(i, 0) - mean_pts_x + mean_ver_x,
                                  -point_3d.coeff(i, 1) - mean_pts_y + mean_ver_y);
        }

        const cv::Point* points[] = {pts[0]};
        int npts[] = {10};
        cv::polylines(img, points, npts,1,true,color,line_width,cv::LINE_AA,0);

        cv::line(img, pts[0][1], pts[0][6], color, line_width, cv::LINE_AA);
        cv::line(img, pts[0][2], pts[0][7], color, line_width, cv::LINE_AA);
        cv::line(img, pts[0][3], pts[0][8], color, line_width, cv::LINE_AA);

    }
#endif
    void Utility::TddfaUtility::load_obj(const char *obj_fp, std::vector<float> &vertices, std::vector<float> &colors, std::vector<int> &triangles, int nver, int ntri) {
        FILE *fp;
        fp = fopen(obj_fp, "r");

        char t; // type: v or f
        if (fp != nullptr) {
            for (int i = 0; i < nver; ++i) {
                fscanf(fp, "%c", &t);
                for (int j = 0; j < 3; ++j){
                    float v;
                    fscanf(fp, " %f", &v);
                    vertices.push_back(v);
                }

                for (int j = 0; j < 3; ++j){
                    float c;
                    fscanf(fp, " %f", &c);
                    colors.push_back(c);
                }
                fscanf(fp, "\n");
            }
//        fscanf(fp, "%c", &t);
            for (int i = 0; i < ntri; ++i) {
                fscanf(fp, "%c", &t);
                for (int j = 0; j < 3; ++j) {
                    int tri;
                    fscanf(fp, " %d", &tri);
                    tri -= 1;
                    triangles.push_back(tri);
                }
                fscanf(fp, "\n");
            }

            fclose(fp);
        }
    }

    void Utility::TddfaUtility::load_obj2(const char *obj, std::vector<float> &vertices, std::vector<float> &colors, std::vector<int> &triangles, int nver, int ntri) {
        std::istringstream iss;
        iss.str(obj);
        std::string line;
        auto nver_rest = nver;
        auto ntri_rest = ntri;

        while (std::getline(iss, line)) {
            if (nver_rest) {
                nver_rest--;

                std::vector<std::string> rst;
                auto split = [&rst](const std::string &str) {
                    std::stringstream ss(str);
                    std::string buffer;
                    while (std::getline(ss, buffer, ' ')) rst.push_back(buffer);
                };
                split(line);
//                assert(rst[0] == 'v');
                for (int i = 0; i < 3; ++i) {
                    vertices.push_back(std::stof(rst[i + 1]));
                    colors.push_back(std::stof(rst[i + 4]));
                }
            }

            if (ntri_rest) {
                ntri_rest--;

                std::vector<std::string> rst;
                auto split = [&rst](const std::string &str) {
                    std::stringstream ss(str);
                    std::string buffer;
                    while (std::getline(ss, buffer, ' ')) rst.push_back(buffer);
                };
                split(line);
//                assert(rst[0] == 'f');
                for (int i = 0; i < 3; ++i) {
                    triangles.push_back(std::stof(rst[i + 1]));
                }
            }

        }
    }


    void Utility::tddfa_post_process(const std::vector<std::vector<float>> &outputs,
                            const std::vector<std::vector<int>> &outputs_shape,
                            const std::shared_ptr<FaceMeta>& face_meta, std::vector<float> &vertices,
                                     std::vector<float> &pose, std::vector<float> &sRt, int target){

        auto param = outputs[0];
//        std::vector<float> p = {param[0], param[1], param[2],
//                                param[4], param[5], param[6],
//                                param[8], param[9], param[10]};
//        std::vector<float> offset = {param[3], param[7], param[11]};

        std::vector<float> vertex(outputs[1].size(), 0);

        int batch_size = outputs_shape[1].size() == 1? 1 : outputs_shape[1][0];
        int pts3d_num = outputs_shape[1][outputs_shape[1].size() - 1] / 3;

        for(int i = 0; i <pts3d_num; i++){
            vertex[i] = (param[0] * outputs[1][i * 3] +  param[1] * outputs[1][i * 3 + 1]+  param[2] * outputs[1][i * 3 + 2]) + param[3];
            vertex[i + pts3d_num] = (param[4] * outputs[1][i * 3] +  param[5] * outputs[1][i * 3 + 1]+  param[6] * outputs[1][i * 3 + 2]) + param[7];
            vertex[i + 2 * pts3d_num] = (param[8] * outputs[1][i * 3] +  param[9] * outputs[1][i * 3 + 1]+  param[10] * outputs[1][i * 3 + 2]) + param[11];
//            std::cout << vertex[i * 3] << "," << vertex[i * 3 + 1] << "," << vertex[i * 3 + 2] << std::endl;
        }
        std::vector<float> pts3d;
        TddfaUtility::similar_transform(vertex, batch_size, pts3d_num, pts3d, face_meta, target);
        vertices.clear();

        for(int n = 0; n < batch_size; n++){
            for(int i = 0; i < pts3d_num; i++){
                auto x = pts3d[n * pts3d_num * 3 + i] + .5; // x
                auto y = pts3d[n * pts3d_num * 3 + pts3d_num + i] + .5; // y
                auto z = pts3d[n * pts3d_num * 3 + 2 * pts3d_num + i] + .5; // z
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            }
        }

        TddfaUtility::calc_pose(outputs[0], sRt, pose);
    }


    void Utility::TddfaUtility::tddfa_rasterize(cv::Mat &image, const std::vector<float> &vertices, const char* obj, int obj_type, bool color_rgb_swap){

        std::vector<float> _vertices;
        std::vector<float> colors;
        std::vector<int> _triangles;

        int nver = 38365;
        int ntri = 76073;

        if(obj_type == 0){
            load_obj(obj, _vertices, colors, _triangles, nver, ntri);
        }
        else{
            load_obj2(obj, _vertices, colors, _triangles, nver, ntri);
        }

        // RGB -> BGR
        if(color_rgb_swap){
            for(int i = 0; i < nver; i ++){
                auto tmp = colors[3 * i];
                colors[3 * i] = colors[3 * i + 2];
                colors[3 * i + 2] = tmp;
            }
        }


    int buffer_size = image.cols * image.rows;
    std::vector<float> depth_buffer(buffer_size, std::numeric_limits<float>::lowest());
    _rasterize(image.data, vertices.data(), triangles.data(), colors.data(), depth_buffer.data(), ntri, image.rows, image.cols, image.channels(), 0.8, false);

    }


#ifndef ENABLE_NCNN_WASM
    void Utility::animated_gan_post_process(const std::vector<float> &output,
                                   const std::vector<int> &outputs_shape, cv::Mat &animated){
        std::vector<uint8_t> _data;

        for_each(output.begin(), output.end(), [=, &_data](float item) {_data.push_back(clip<uint8_t>((item + 1.f)  * 127.5f, 0, 255));});

        std::vector<uint8_t> _rgb;
        _rgb.resize(outputs_shape[1] * outputs_shape[2] * outputs_shape[3]);
        for(int c = 0; c < outputs_shape[1]; c++){
            for(int y = 0; y < outputs_shape[2]; y++){
                for(int x = 0; x < outputs_shape[3]; x++){
                    _rgb[y * outputs_shape[3] * outputs_shape[1] + x * outputs_shape[1] + c] = _data[c * outputs_shape[2] * outputs_shape[3] + y * outputs_shape[3] + x];
                }
            }
        }


        cv::Mat _result(outputs_shape[2], outputs_shape[3], CV_8UC3, _rgb.data());

        cv::cvtColor(_result, animated, cv::COLOR_BGR2RGB);

    }

    void Utility::bisenet_post_process(const cv::Mat &src_image, cv::Mat &parsing_image, const std::vector<float> &outputs,
                              const std::vector<int> &outputs_shape, bool fushion, const std::vector<int>& ignore){

        int dim = outputs_shape.size();
        int output_n = dim == 3? 1: outputs_shape[0];
        // only support 1 batch
//        assert(1 == output_n);
        int output_c = outputs_shape[outputs_shape.size() - 3];
        int output_h = outputs_shape[outputs_shape.size() - 2];
        int output_w = outputs_shape[outputs_shape.size() - 1];
        std::vector<int> parsing_anno;

        parsing_anno.resize(output_h * output_w);

        for(int h = 0; h < output_h; h++){
            for(int w = 0; w < output_w; w++){

                int max_index = -1;
                float max_value = std::numeric_limits<float>::lowest();
                for(int c = 0; c < output_c; c++){
                    int cur_index = c * output_h * output_w + h * output_w + w;
                    if(outputs[cur_index] >= max_value){
                        max_index = c;
                        max_value = outputs[cur_index];
                    }
                }

                parsing_anno[h * output_w + w] = max_index;
            }
        }
        cv::Mat vis_parsing_anno_color(output_h, output_w, CV_8UC3, cv::Scalar::all(255));

        for(int i = 0; i < vis_parsing_anno_color.rows; i++){
            for(int j = 0; j < vis_parsing_anno_color.cols; j++){
                bool skip = false;
                if(parsing_anno[i * output_w + j] > 0 && parsing_anno[i * output_w + j] < 20) {
                    if(!ignore.empty()){
                        for(auto ign: ignore){
                            if(parsing_anno[i * output_w + j] == ign){
                                skip = true;
                                continue;
                            }
                        }
                    }
                    if(skip)
                        continue;
//                    if(parsing_anno[i * output_w + j] == 18)
//                        continue;
//                    if(parsing_anno[i * output_w + j] == 16)
//                        continue;
                    auto color = parsing_part_colors[parsing_anno[i * output_w + j]];
                    vis_parsing_anno_color.at<cv::Vec3b>(i,j) = cv::Vec3b(color[0], color[1], color[2]);
                }

            }

        }
        if(fushion){
            cv::Mat resized = src_image;
            if(src_image.cols != output_w || src_image.rows != output_h){
                cv::resize(src_image, resized, cv::Size(output_w, output_h));
            }
            cv::addWeighted(resized, 0.4, vis_parsing_anno_color, 0.6, 0, parsing_image);
        } else{
            parsing_image = vis_parsing_anno_color.clone();
        }


    }



    void Utility::stylegan_post_process(cv::Mat &result_image, const std::vector<float> &output,
                                        const std::vector<int> &output_shape) {

        cv::Mat generated_image(output_shape[2], output_shape[3], CV_8UC3);
        auto generated_ptr = output.data();
        for(int h = 0; h < output_shape[2]; h++){
            for(int w = 0; w < output_shape[3]; w++){
                for(int c = 0; c < output_shape[1]; c++){
                    float item = generated_ptr[c * output_shape[2] * output_shape[3] + h * output_shape[3] + w];
                    auto pixel = clip<uchar>((clip<float>(item, -1, 1) + 1.0f) / 2.0f * 255.0f + .5f, 0, 255);
                    generated_image.data[h * output_shape[3] * output_shape[1] + output_shape[1] * w + c] = pixel;
                }
            }
        }

        cv::cvtColor(generated_image, result_image, cv::COLOR_BGR2RGB);

    }

    // TPSMM

    double Utility::hull_area(const std::vector<float>& pts, float epsilon, bool closed){
        std::vector<cv::Point2f> hull;
        std::vector<cv::Point2f> contour;
        std::vector<cv::Point2f> points;
        points.reserve(pts.size() / 2);
        for(int i = 0; i < pts.size() / 2; i++){
            points.emplace_back(pts[2 * i], pts[2 * i + 1]);
        }
        cv::convexHull(cv::Mat(points), hull);
        cv::approxPolyDP(cv::Mat(hull), contour, epsilon, closed);
        return cv::contourArea(cv::Mat(contour));
    }

    int Utility::relative_kp(const std::vector<float>& kp_source,
                    const std::vector<float>& kp_driving,
                    const std::vector<float>& kp_driving_initial,
                    std::vector<float>& kp_norm){


        auto source_area = hull_area(kp_source);
        auto driving_area = hull_area(kp_driving_initial);
        auto adapt_movement_scale = sqrt(source_area) / sqrt(driving_area);

        kp_norm.clear();
        kp_norm.resize(kp_source.size(), 0);
        for(int i = 0; i < kp_source.size(); i++){
            kp_norm[i] = (kp_driving[i] - kp_driving_initial[i]) * adapt_movement_scale + kp_source[i];
        }

        return 0;
    }


#endif
    void Utility::yolov7_post_process(const std::vector<std::vector<float>> &outputs,
                             const std::vector<std::vector<int>> &outputs_shape,
                             std::vector<std::shared_ptr<ObjectMeta>> &results,
                             float conf_thresh, float nms_thresh,
                             float scale){

        static float stride[3] = {8.0, 16.0, 32.0};
        static float anchor_grid[3][3][2] = {
                {{12.,  16.}, {19.,  36.}, {40.,  28.} },
                {{36.,  75.}, {76.,  55.}, {72., 146.} },
                {{142., 110.}, {192., 243.}, {459., 401.} }
        };

        int nb = 0;
        int no = outputs_shape[0][4];
        for(auto shape: outputs_shape){
            nb += (shape[1] * shape[2] * shape[3]);
        }

        std::vector<float> output;
        std::vector<int> output_shape{1, nb, no};
        output.resize(nb * no);

        for(int i = 0; i < outputs.size(); i++){
//            assert(1 == outputs_shape[i][0]);

            int na = outputs_shape[i][1];
            int ny = outputs_shape[i][2];
            int nx = outputs_shape[i][3];

            for(int a = 0; a < na; a++){
                for (int y = 0; y < ny; y++){
                    for (int x = 0; x < nx; x++){
                        memcpy(output.data() + a * nx * ny * no + y * nx * no + x * no,
                               outputs[i].data() + a * nx * ny * no + y * nx * no + x * no, no * sizeof(float));
                        output[a * nx * ny * no + y * nx * no + x * no] = outputs[i][a * nx * ny * no + y * nx * no + x * no] * (2 * stride[i]) + stride[i] * (x -.5f);
                        output[a * nx * ny * no + y * nx * no + x * no + 1] = outputs[i][a * nx * ny * no + y * nx * no + x * no + 1] * (2 * stride[i]) + stride[i] * (y -.5f);;
                        output[a * nx * ny * no + y * nx * no + x * no + 2] = pow(outputs[i][a * nx * ny * no + y * nx * no + x * no + 2], 2) * 4 * anchor_grid[i][a][0];
                        output[a * nx * ny * no + y * nx * no + x * no + 3] = pow(outputs[i][a * nx * ny * no + y * nx * no + x * no + 3], 2) * 4 * anchor_grid[i][a][1];

                    }
                }
            }
        }

        AIDB::Utility::yolov7_post_process(output, output_shape, results, conf_thresh, nms_thresh, scale);
    }

    void Utility::yolov7_post_process(const std::vector<float> &output,
                                      const std::vector<int> &output_shape,
                                      std::vector<std::shared_ptr<ObjectMeta>> &results,
                                      float conf_thresh, float nms_thresh,
                                      float scale) {

        results.clear();
//        assert(3 == output_shape.size());
//        assert(1 == output_shape[0]);
        int bs = output_shape[0];  // batch size 1 here
        int nc = output_shape[2] - 5;  // number of classes
        int np = output_shape[1];

        for(int n = 0; n < bs; n++) {
            std::vector<std::shared_ptr<ObjectMeta>> object;
            for (int i = 0; i < np; i++) {
                float cls_conf = output[n * output_shape[2] *np + i * output_shape[2] + 4];

                int max_index = -1;
                float max_obj_conf = std::numeric_limits<float>::lowest();
                for (int j = 0; j < nc; j++) {
                    auto obj_conf = output[n * output_shape[2] *np + i * output_shape[2] + 5 + j];
                    if (obj_conf >= max_obj_conf) {
                        max_obj_conf = obj_conf;
                        max_index = j;
                    }
                }

                float conf = max_obj_conf * cls_conf;

                if (conf < conf_thresh) {
                    continue;
                }

                std::shared_ptr<ObjectMeta> obj = std::make_shared<ObjectMeta>();

                obj->x1 = output[n * output_shape[2] *np + i * output_shape[2]] - output[n * output_shape[2] *np + i * output_shape[2] + 2] / 2;
                obj->y1 = output[n * output_shape[2] *np + i * output_shape[2] + 1] - output[n * output_shape[2] *np + i * output_shape[2] + 3] / 2;
                obj->x2 = obj->x1 + output[n * output_shape[2] *np + i * output_shape[2] + 2];
                obj->y2 = obj->y1 + output[n * output_shape[2] *np + i * output_shape[2] + 3];

                obj->score = conf;
                obj->label = max_index;

                object.emplace_back(obj);
            }

            Common::NMS<ObjectMeta>(object, results, nms_thresh);

            for(auto& obj: results){
                obj->x1 /= scale;
                obj->y1 /= scale;
                obj->x2 /= scale;
                obj->y2 /= scale;

            }
        }
    }

//    void Utility::yolov8_post_process(const std::vector<float> &output,
//                             const std::vector<int> &output_shape,
//                             std::vector<std::shared_ptr<ObjectMeta>> &results,
//                             float conf_thresh, float nms_thresh,
//                             float scale){
//
//        results.clear();
//
//        assert(1 == output_shape[0]);
//
//        int bs = output_shape[0];  // batch size 1 here
//        int nc = output_shape[1] - 4;  // number of classes
//        int na = output_shape[2];
//
//        for(int n = 0; n < bs; n++) {
//            std::vector<std::shared_ptr<ObjectMeta>> object;
//            for (int i = 0; i < na; i++) {
//                int max_index = -1;
//                float max_conf = std::numeric_limits<float>::lowest();
//                for (int j = 0; j < nc; j++) {
//                    auto cur_conf = output[n * (nc + 4) * na + (j + 4) * na + i];
//                    if (cur_conf >= max_conf) {
//                        max_conf = cur_conf;
//                        max_index = j;
//                    }
//                }
//
//                if (max_conf < conf_thresh) {
//                    continue;
//                }
//
//                std::shared_ptr<ObjectMeta> obj = std::make_shared<ObjectMeta>();
//
//                obj->x1 = output[n * (nc + 4) * na + i] - output[n * (nc + 4) * na + 2 * na + i] / 2;
//                obj->y1 = output[n * (nc + 4) * na + na + i] - output[n * (nc + 4) * na + 3 * na + i] / 2;
//                obj->x2 = obj->x1 + output[n * (nc + 4) * na + 2 * na + i];
//                obj->y2 = obj->y1 + output[n * (nc + 4) * na + 3 * na + i];
//
//                obj->score = max_conf;
//                obj->label = max_index;
//
//                object.emplace_back(obj);
//            }
//
//            Common::NMS<ObjectMeta>(object, results, nms_thresh);
//
//            for(auto& obj: results){
//                obj->x1 /= scale;
//                obj->y1 /= scale;
//                obj->x2 /= scale;
//                obj->y2 /= scale;
//
//            }
//        }
//    }

    void Utility::yolov8_post_process(const std::vector<float> &output,
                                      const std::vector<int> &output_shape,
                                      std::vector<std::shared_ptr<ObjectMeta>> &results,
                                      float conf_thresh, float nms_thresh,
                                      float scale){

        results.clear();

//        assert(1 == output_shape[0]);

        static int strides[3] = {8, 16, 32};
        static int input_size = 640;
        int offset = 0;

        int bs = output_shape[0];  // batch size 1 here
        int nc = output_shape[1] - 4;  // number of classes
        int na = output_shape[2];

        for(int n = 0; n < bs; n++) {
            std::vector<std::shared_ptr<ObjectMeta>> object;
            for(auto stride: strides){
                int nx, ny;
                nx = ny = input_size / stride;
                for(int y = 0; y < ny; y++){
                    for(int x = 0; x < nx; x++){
                        int max_index = -1;
                        float max_conf = std::numeric_limits<float>::lowest();
                        for (int j = 0; j < nc; j++) {
                            auto cur_conf = output[n * (nc + 4) * na + offset + y * nx + x + (4 + j) * na];
                            if (cur_conf >= max_conf) {
                                max_conf = cur_conf;
                                max_index = j;
                            }
                        }

                        if (max_conf < conf_thresh) {
                            continue;
                        }

                        std::shared_ptr<ObjectMeta> obj = std::make_shared<ObjectMeta>();

                        obj->x1 = (x + .5 - output[n * (nc + 4) * na + offset + y * nx + x]) * stride;
                        obj->y1 = (y + .5 - output[n * (nc + 4) * na + offset + y * nx + x + na]) * stride;
                        obj->x2 = (x + .5 + output[n * (nc + 4) * na + offset + y * nx + x + 2 * na]) * stride;
                        obj->y2 = (y + .5 + output[n * (nc + 4) * na + offset + y * nx + x + 3 * na]) * stride;

                        obj->score = max_conf;
                        obj->label = max_index;

                        object.emplace_back(obj);
                    }
                }
                offset += (nx * ny);
            }


            Common::NMS<ObjectMeta>(object, results, nms_thresh);

            for(auto& obj: results){
                obj->x1 /= scale;
                obj->y1 /= scale;
                obj->x2 /= scale;
                obj->y2 /= scale;

            }
        }

    }

    void Utility::mobile_sam_post_process(const std::vector<float> &output,
                                          const cv::Mat &source,
                                          cv::Mat &result,
                                          float scale,
                                          const cv::Scalar &color,
                                          float alpha) {
        cv::Mat Mask(256, 256, CV_32FC1, (void*)output.data());

        cv::resize(Mask, Mask, cv::Size(1024, 1024));
        cv::Mat m_roi = Mask(cv::Range(0, source.rows * scale),
                             cv::Range(0, source.cols * scale));
        cv::resize(m_roi, m_roi, cv::Size(source.cols, source.rows));

        cv::Mat binary;
        cv::threshold(m_roi, binary, 0.0, 1.0, cv::THRESH_BINARY);

        std::vector<cv::Mat> mv;

        cv:: Mat b = binary * color[0];
        cv:: Mat g = binary * color[1];
        cv:: Mat r = binary * color[2];
        mv.push_back(b);
        mv.push_back(g);
        mv.push_back(r);

        cv::merge(mv,Mask);

        Mask.convertTo(Mask, CV_8U);

        cv::Mat _result1;
        binary = (binary * 255);
        binary.convertTo(binary, CV_8U);

        cv::bitwise_and(source, source, _result1, binary);

        cv::Mat _result2;

        cv::bitwise_and(source, source, _result2, 255 - binary);

        result = _result2 + 0.5 * _result1 + 0.5 * Mask;

    }


    // MoveNet
    int Utility::MoveNetUtility::line_map[20][2] = {{2,  1}, {2, 4}, {1, 3}, {4, 0}, {0, 3},
                                            {4,  6}, {3, 5}, {6, 8}, {8, 10}, {5, 7}, {7, 9},
                                            {6,  12}, {5, 11}, {12, 11}, {12, 14}, {11, 13},
                                            {14, 16}, {13, 15}, {2, 0}, {1, 0}};

    void Utility::MoveNetUtility::MoveNetDecode(const vector<std::vector<float>> &outputs,
                                                const vector<std::vector<int>> &outputs_shape,
                                                vector<std::vector<float>> &decoded_keypoints, int joints_num,
                                                float heatmap_thresh, int target_size) {

//        assert(4 == outputs.size());
//        assert(4 == outputs_shape.size());

        decoded_keypoints.clear();

        std::vector<float> heatmaps = outputs[0];
        std::vector<float> centers= outputs[1];
        std::vector<float> regs = outputs[2];
        std::vector<float> offsets = outputs[3];

        int heatmap_width = outputs_shape[0][outputs_shape[0].size() - 1];
        int heatmap_height = outputs_shape[0][outputs_shape[0].size() - 2];

        // heatmap
        for_each(heatmaps.begin(), heatmaps.end(), [heatmap_thresh](float &item){item = item<heatmap_thresh?0:item;});

        float cx, cy;
        get_max_points(centers, heatmap_height, heatmap_width, 1, cx, cy);

        for(int n = 0; n < joints_num; n++){

            float reg_x_origin = regs[2 * n * heatmap_width * heatmap_height + cy * heatmap_width + cx] + .5f;
            float reg_y_origin = regs[(2 * n + 1) * heatmap_width * heatmap_height + cy * heatmap_width + cx] + .5f;

            float reg_x = reg_x_origin + cx;
            float reg_y = reg_y_origin + cy;

            // 构造关键点坐标最小、周边递增的权重矩阵，目的是取出最靠近中心点的人的关键点
            std::vector<float> tmp_reg(heatmap_height * heatmap_width, 0);
            tmp_reg.resize(heatmap_height * heatmap_width);

            for(int y = 0; y < heatmap_height; y++){
                for(int x = 0; x < heatmap_width; x++){
                    float tmp_weight = sqrt(pow((x - reg_x), 2) + pow((y - reg_y), 2)) + 1.8f;
                    tmp_reg[y * heatmap_width + x] = heatmaps[n * heatmap_width *  heatmap_height + y * heatmap_width + x] / tmp_weight;
                }
            }
            float kx, ky;
            get_max_points(tmp_reg, heatmap_height, heatmap_width, 1, kx, ky, false);

            kx = clip<float>(kx, 0, 47);
            ky = clip<float>(ky, 0, 47);

            float score = heatmaps[n * heatmap_width *  heatmap_height + ky * heatmap_width + kx];


            float offset_x = offsets[2 * n * heatmap_width * heatmap_height + ky * heatmap_width + kx];
            float offset_y = offsets[(2 * n + 1)  * heatmap_width * heatmap_height + ky * heatmap_width + kx];

            kx = (kx + offset_x) / target_size;
            ky = (ky + offset_y) / target_size;


            kx = kx < heatmap_thresh? -1: kx;
            ky = ky < heatmap_thresh? -1: ky;

            decoded_keypoints.push_back({kx, ky});

        }

    }

    void Utility::MoveNetUtility::get_max_points(const vector<float> &output, int height, int width, int channel,
                                                 float &max_x, float &max_y, bool center) {
        std::vector<float> output_cpy(output.begin(), output.end());

        float max_value = std::numeric_limits<float>::lowest();
        if(center){
            for(int c = 0; c < channel; c++){
                for(int h = 0; h < height; h++){
                    for(int w = 0; w < width; w++){
                        output_cpy[c * height * width + h * width + w] *= center_weight[h][w];
                        if(output_cpy[c * height * width + h * width + w] >= max_value){
                            max_value = output_cpy[c * height * width + h * width + w];
                            max_x = w;
                            max_y = h;
                        }
                    }
                }
            }
        } else{
            for(int c = 0; c < channel; c++){
                for(int h = 0; h < height; h++){
                    for(int w = 0; w < width; w++){
                        if(output_cpy[c * height * width + h * width + w] >= max_value){
                            max_value = output_cpy[c * height * width + h * width + w];
                            max_x = w;
                            max_y = h;
                        }
                    }
                }
            }
        }

    }


    void Utility::YoloX::generate_grids_and_stride(int target_size, const vector<int> &strides) {
        for (auto stride : strides){
            int num_grid = target_size / stride;
            for (int g1 = 0; g1 < num_grid; g1++){
                for (int g0 = 0; g0 < num_grid; g0++){
                    GridAndStride gs{};
                    gs.grid0 = g0;
                    gs.grid1 = g1;
                    gs.stride = stride;
                    _grid_strides.push_back(gs);
                }
            }
        }
    }

    void Utility::YoloX::generate_proposals(const vector<float> &feat_blob,
                                             const vector<int> &outputs_shape, float prob_threshold,
                                            std::vector<std::shared_ptr<ObjectMeta>> &objects) {
        const int num_grid = outputs_shape[1];
        const int num_class = outputs_shape[2] - 5;
        const int num_anchors = _grid_strides.size();

        const float* feat_ptr = feat_blob.data();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
        {
            const int grid0 = _grid_strides[anchor_idx].grid0;
            const int grid1 = _grid_strides[anchor_idx].grid1;
            const int stride = _grid_strides[anchor_idx].stride;

            float x_center = (feat_ptr[0] + grid0) * stride;
            float y_center = (feat_ptr[1] + grid1) * stride;
            float w = exp(feat_ptr[2]) * stride;
            float h = exp(feat_ptr[3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            float box_objectness = feat_ptr[4];
            for (int class_idx = 0; class_idx < num_class; class_idx++){
                float box_cls_score = feat_ptr[5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > prob_threshold){
                    std::shared_ptr<ObjectMeta> obj = std::make_shared<ObjectMeta>();
                    obj->x1 = x0;
                    obj->y1 = y0;
                    obj->x2 = x0 + w;
                    obj->y2 = y0 + h;
                    obj->label = class_idx;
                    obj->score = box_prob;
                    objects.emplace_back(obj);
                }

            } // class loop
            feat_ptr += outputs_shape[2];

        } // point anchor loop

    }

    void Utility::YoloX::forward(const std::vector<float>& outputs, const std::vector<int>& outputs_shape, std::vector<std::shared_ptr<ObjectMeta>> &result, int src_width, int src_height, float scale) {

        std::vector<std::shared_ptr<ObjectMeta>> proposals;
        generate_proposals(outputs, outputs_shape, _conf_thresh, proposals);
        sort(proposals.begin(), proposals.end(), [](const std::shared_ptr<ObjectMeta>& obj1,const std::shared_ptr<ObjectMeta>& obj2){ return obj1->score > obj2->score;});
        result.clear();
        Common::NMS<ObjectMeta>(proposals, result, _nms_thresh);
        for(auto &obj: result){
            float x1 = obj->x1 / scale;
            float y1 = obj->y1 / scale;
            float x2 = obj->x2 / scale;
            float y2 = obj->y2 / scale;

            // clip
            x1 = std::max(std::min(x1, (float)(src_width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(src_height - 1)), 0.f);
            x2 = std::max(std::min(x2, (float)(src_width - 1)), 0.f);
            y2 = std::max(std::min(y2, (float)(src_height - 1)), 0.f);

            obj->x1 = x1;
            obj->y1 = y1;
            obj->x2 = x2;
            obj->y2 = y2;
        }
    }

    void Utility::YoloX::operator()(const vector<float> &outputs, const vector<int> &outputs_shape,
                                    vector<std::shared_ptr<ObjectMeta>> &result, int src_width, int src_height,
                                    float scale) {
        forward(outputs, outputs_shape, result, src_width, src_height, scale);

    }

#ifndef ENABLE_NCNN_WASM
    std::vector<std::vector<float>> Utility::PPOCR::Mat2Vector(cv::Mat mat) {
        std::vector<std::vector<float>> img_vec;
        std::vector<float> tmp;

        for (int i = 0; i < mat.rows; ++i) {
            tmp.clear();
            for (int j = 0; j < mat.cols; ++j) {
                tmp.push_back(mat.at<float>(i, j));
            }
            img_vec.push_back(tmp);
        }
        return img_vec;
    }


    std::vector<std::vector<float>>
    Utility::PPOCR::GetMiniBoxes(cv::RotatedRect box, float &ssid) {
        ssid = std::max(box.size.width, box.size.height);

        cv::Mat points;
        cv::boxPoints(box, points);

        auto array = Mat2Vector(points);
        std::sort(array.begin(), array.end(), [](const std::vector<float> &a, const std::vector<float> &b){ return
                a[0] != b[0] && a[0] < b[0];});

        std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                idx4 = array[3];
        if (array[3][1] <= array[2][1]) {
            idx2 = array[3];
            idx3 = array[2];
        } else {
            idx2 = array[2];
            idx3 = array[3];
        }
        if (array[1][1] <= array[0][1]) {
            idx1 = array[1];
            idx4 = array[0];
        } else {
            idx1 = array[0];
            idx4 = array[1];
        }

        array[0] = idx1;
        array[1] = idx2;
        array[2] = idx3;
        array[3] = idx4;

        return array;
    }

    void Utility::PPOCR::BoxesFromBitmap(
            const cv::Mat& pred, const cv::Mat& bitmap,
            std::vector<std::shared_ptr<OcrMeta>> &det_results){

        int width = bitmap.cols;
        int height = bitmap.rows;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                         cv::CHAIN_APPROX_SIMPLE);

        int num_contours = contours.size() >= _max_candidates ? _max_candidates : contours.size();

        for (int _i = 0; _i < num_contours; _i++) {
            if (contours[_i].size() <= 2) {
                continue;
            }
            float ssid;
            cv::RotatedRect box = cv::minAreaRect(contours[_i]);
            auto array = GetMiniBoxes(box, ssid);

            const auto& box_for_unclip = array;
            // end get_mini_box

            if (ssid < _min_size) {
                continue;
            }

            float score;
            if (_det_db_score_mode == "slow")
                /* compute using polygon*/
                score = PolygonScoreAcc(contours[_i], pred);
            else
                score = BoxScoreFast(array, pred);

            if (score < _det_db_box_thresh)
                continue;

            // start for unclip
            cv::RotatedRect points;
            UnClip(box_for_unclip, _det_db_unclip_ratio, points);
            if (points.size.height < 1.001 && points.size.width < 1.001) {
                continue;
            }
            // end for unclip

            cv::RotatedRect clipbox = points;
            auto cliparray = GetMiniBoxes(clipbox, ssid);

            if (ssid < _min_size + 2)
                continue;

            int dest_width = pred.cols;
            int dest_height = pred.rows;
            std::shared_ptr<OcrMeta> intcliparray = std::make_shared<OcrMeta>();

            for (int num_pt = 0; num_pt < 4; num_pt++) {

                PointT<int> a = PointT<int>(int(clip<float>(roundf(cliparray[num_pt][0] / float(width) *
                                                           float(dest_width)),
                                                    0, float(dest_width))),
                                    int(clip<float>(roundf(cliparray[num_pt][1] /
                                                           float(height) * float(dest_height)),
                                                    0, float(dest_height))));
                intcliparray->box.push_back(a);

            }
            det_results.push_back(intcliparray);

        } // end for
    }

    void Utility::PPOCR::dbnet_post_process(const std::vector<float>& output, const std::vector<int>& output_shape,
                                            std::vector<std::shared_ptr<OcrMeta>> &det_results,
                                            float ratio_h, float ratio_w, const cv::Mat& srcimg){
        int n2 = output_shape[2];
        int n3 = output_shape[3];
        int n = n2 * n3;
        std::vector<float> pred(n, 0.0);
        std::vector<unsigned char> cbuf(n, ' ');

        for (int i = 0; i < n; i++) {
            pred[i] = float(output[i]);
            cbuf[i] = (unsigned char)((output[i]) * 255);
        }
        cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
        cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());


        cv::Mat bit_map;
        cv::threshold(cbuf_map, bit_map, _det_db_thresh * 255, 255, cv::THRESH_BINARY);
        if (_use_dilation) {
            cv::Mat dila_ele =
                    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::dilate(bit_map, bit_map, dila_ele);
        }

        std::vector<std::shared_ptr<OcrMeta>> boxes;
        BoxesFromBitmap(pred_map, bit_map, boxes);
        FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg, det_results);
    }

    float Utility::PPOCR::PolygonScoreAcc(const std::vector<cv::Point> &contour, const cv::Mat &pred) {

        int width = pred.cols;
        int height = pred.rows;
        std::vector<float> box_x;
        std::vector<float> box_y;
        for (auto & c : contour) {
            box_x.push_back(c.x);
            box_y.push_back(c.y);
        }

        auto xmin = clip<int>(std::floor(*(std::min_element(box_x.begin(), box_x.end()))), 0,width - 1);
        auto xmax = clip<int>(std::ceil(*(std::max_element(box_x.begin(), box_x.end()))), 0, width - 1);
        auto ymin = clip<int>(std::floor(*(std::min_element(box_y.begin(), box_y.end()))), 0, height - 1);
        auto ymax = clip<int>(std::ceil(*(std::max_element(box_y.begin(), box_y.end()))), 0, height - 1);

        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

        auto *rook_point = new cv::Point[contour.size()];

        for (int i = 0; i < contour.size(); ++i) {
            rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
        }
        const cv::Point *ppt[1] = {rook_point};
        int npt[] = {int(contour.size())};

        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

        cv::Mat croppedImg;
        pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
                .copyTo(croppedImg);
        float score = cv::mean(croppedImg, mask)[0];

        delete[] rook_point;
        return score;
    }

    float Utility::PPOCR::BoxScoreFast(const std::vector<std::vector<float>>& box_array, const cv::Mat &pred) {
        auto array = box_array;
        int width = pred.cols;
        int height = pred.rows;

        float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
        float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

        auto xmin = clip<int>(std::floor(*(std::min_element(box_x, box_x + 4))), 0, width - 1);
        auto xmax = clip<int>(std::ceil(*(std::max_element(box_x, box_x + 4))), 0,width - 1);
        auto ymin = clip<int>(std::floor(*(std::min_element(box_y, box_y + 4))), 0,height - 1);
        auto ymax = clip<int>(std::ceil(*(std::max_element(box_y, box_y + 4))), 0,height - 1);

        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

        cv::Point root_point[4];
        root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
        root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
        root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
        root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
        const cv::Point *ppt[1] = {root_point};
        int npt[] = {4};
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

        cv::Mat croppedImg;
        pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
                .copyTo(croppedImg);

        auto score = cv::mean(croppedImg, mask)[0];
        return score;
    }

    void Utility::PPOCR::UnClip(const vector<std::vector<float>> &box, float unclip_ratio, cv::RotatedRect &rotate_rect) {
        float distance = 1.0;

        GetContourArea(box, unclip_ratio, distance);

        ClipperLib::ClipperOffset offset;
        ClipperLib::Path p;
        p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
          << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
          << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
          << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
        offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

        ClipperLib::Paths soln;
        offset.Execute(soln, distance);
        std::vector<cv::Point2f> points;

        for (int j = 0; j < soln.size(); j++) {
            for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
                points.emplace_back(soln[j][i].X, soln[j][i].Y);
            }
        }

        if (points.empty()) {
            rotate_rect = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
        } else {
            rotate_rect = cv::minAreaRect(points);
        }

    }

    void Utility::PPOCR::GetContourArea(const vector<std::vector<float>> &box, float unclip_ratio, float &distance) {
        int pts_num = 4;
        float area = 0.0f;
        float dist = 0.0f;
        for (int i = 0; i < pts_num; i++) {
            area += box[i][0] * box[(i + 1) % pts_num][1] -
                    box[i][1] * box[(i + 1) % pts_num][0];
            dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                          (box[i][0] - box[(i + 1) % pts_num][0]) +
                          (box[i][1] - box[(i + 1) % pts_num][1]) *
                          (box[i][1] - box[(i + 1) % pts_num][1]));
        }
        area = fabs(float(area / 2.0));

        distance = area * unclip_ratio / dist;
    }

    void Utility::PPOCR::FilterTagDetRes(std::vector<std::shared_ptr<OcrMeta>> boxes, float ratio_h,
                         float ratio_w, const cv::Mat& srcimg, std::vector<std::shared_ptr<OcrMeta>> &det_results){
        int oriimg_h = srcimg.rows;
        int oriimg_w = srcimg.cols;

        for(auto &box: boxes){
            box = OrderPointsClockwise(box);
            for (auto & m : box->box) {
                m._x /= ratio_w;
                m._y /= ratio_h;

                m._x = int(min(max(m._x, 0), oriimg_w - 1));
                m._y = int(min(max(m._y, 0), oriimg_h - 1));
            }
        }

        for (auto & box : boxes) {
            int rect_width, rect_height;
            rect_width = int(sqrt(pow(box->box[0]._x - box->box[1]._x, 2) +
                                  pow(box->box[0]._y - box->box[1]._y, 2)));
            rect_height = int(sqrt(pow(box->box[0]._x - box->box[3]._x, 2) +
                                   pow(box->box[0]._y - box->box[3]._y, 2)));
            if (rect_width <= 4 || rect_height <= 4)
                continue;
            det_results.push_back(box);
        }

    }

    std::shared_ptr<OcrMeta> Utility::PPOCR::OrderPointsClockwise(const std::shared_ptr<OcrMeta> &pts) {

        const std::shared_ptr<OcrMeta>& box = pts;
        std::sort(box->box.begin(), box->box.end(), [](const PointT<int> &a, const PointT<int> &b){ return
                a._x != b._x && a._x < b._x;});

        PointT<int> leftmost0 = box->box[0];
        PointT<int> leftmost1 = box->box[1];

        PointT<int> rightmost0 = box->box[2];
        PointT<int> rightmost1 = box->box[3];

        std::shared_ptr<OcrMeta> meta = std::make_shared<OcrMeta>();

        meta->box.resize(4);
        if(leftmost0._y > leftmost1._y){
            meta->box[0] = leftmost1;
            meta->box[3] = leftmost0;

        } else{

            meta->box[0] = leftmost0;
            meta->box[3] = leftmost1;
        }

        if(rightmost0._y > rightmost1._y){
            meta->box[1] = rightmost1;
            meta->box[2] = rightmost0;
        } else{
            meta->box[1] = rightmost0;
            meta->box[2] = rightmost1;
        }

        return meta;
    }

    void Utility::PPOCR::GetRotateCropImage(const cv::Mat &src, cv::Mat &dst, const std::shared_ptr<OcrMeta> &meta) {
        cv::Mat image;
        src.copyTo(image);
        std::shared_ptr<OcrMeta> points = std::make_shared<OcrMeta>();
        points->box.assign(meta->box.begin(), meta->box.end());
        points->conf = meta->conf;
        points->label = meta->label;
        points->conf_rotate = meta->conf_rotate;

        int x_collect[4] = {meta->box[0]._x, meta->box[1]._x, meta->box[2]._x, meta->box[3]._x};
        int y_collect[4] = {meta->box[0]._y, meta->box[1]._y, meta->box[2]._y, meta->box[3]._y};
        int left = int(*std::min_element(x_collect, x_collect + 4));
        int right = int(*std::max_element(x_collect, x_collect + 4));
        int top = int(*std::min_element(y_collect, y_collect + 4));
        int bottom = int(*std::max_element(y_collect, y_collect + 4));

        cv::Mat img_crop;
        image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

        for (auto & p : points->box) {
            p._x -= left;
            p._y -= top;
        }

        int img_crop_width = int(sqrt(pow(points->box[0]._x - points->box[1]._x, 2) +
                                      pow(points->box[0]._y - points->box[1]._y, 2)));
        int img_crop_height = int(sqrt(pow(points->box[0]._x - points->box[3]._x, 2) +
                                       pow(points->box[0]._y - points->box[3]._y, 2)));

        cv::Point2f pts_std[4];
        pts_std[0] = cv::Point2f(0., 0.);
        pts_std[1] = cv::Point2f(img_crop_width, 0.);
        pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
        pts_std[3] = cv::Point2f(0.f, img_crop_height);

        cv::Point2f pointsf[4];
        pointsf[0] = cv::Point2f(points->box[0]._x, points->box[0]._y);
        pointsf[1] = cv::Point2f(points->box[1]._x, points->box[1]._y);
        pointsf[2] = cv::Point2f(points->box[2]._x, points->box[2]._y);
        pointsf[3] = cv::Point2f(points->box[3]._x, points->box[3]._y);

        cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

        cv::warpPerspective(img_crop, dst, M,
                            cv::Size(img_crop_width, img_crop_height),
                            cv::BORDER_REPLICATE);

        if (float(dst.rows) >= float(dst.cols) * 1.5) {
            cv::transpose(dst, dst);
            cv::flip(dst, dst, 0);
        }
    }


    void Utility::PPOCR::ReadDict(const std::string &path, std::vector<std::string>& m_vec) {
        std::ifstream in(path);
        std::string line;

        if (in) {
            while (getline(in, line)) {
                m_vec.push_back(line);
            }
        } else {
            std::cout << "no such label file: " << path << ", exit the program..."
                      << std::endl;
            exit(1);
        }
    }

    void Utility::PPOCR::cls_post_process(const std::vector<float>& output, const std::vector<int>& output_shape,
                                          const cv::Mat& src, cv::Mat& dst, std::shared_ptr<AIDB::OcrMeta> ocr_result) {

        assert(2 == output.size());
        ocr_result->conf_rotate = output[1];
        // 上下镜像翻转
        if(output[1] > _cls_thresh){
            cv::rotate(src, dst, 1);
        }

    }

    void Utility::PPOCR::crnn_post_process(const vector<float> &output, const vector<int> &output_shape, std::shared_ptr<AIDB::OcrMeta> ocr_result) {
        std::string str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;

        for (int m = 0; m < output_shape[1]; m++) {
            float max_value = 0.0f;
            for (int nr = 0; nr < output_shape[2]; nr++) {

                if(output[output_shape[2] * m + nr] >= max_value){
                    max_value = output[output_shape[2] * m + nr];
                    argmax_idx = nr;
                }
            }

            if (argmax_idx > 0 && (!(m > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                str_res += _label_list[argmax_idx];
            }
            last_index = argmax_idx;
        }

        score /= count;
        if (std::isnan(score)) {
            ocr_result->conf = -1;
        } else {
            ocr_result->label = str_res;
            ocr_result->conf = score;

        }
    }

    void Utility::PPOCR::draw_objects(cv::Mat &image, const std::shared_ptr<OcrMeta> &meta, const std::string fontFileName){
        cv::Point rook_points[4];
        int min_x = image.cols;
        int min_y = image.rows;
        int max_y = 0;
        for (int m = 0; m < meta->box.size(); m++) {
            rook_points[m] =
                    cv::Point(int(meta->box[m]._x), int(meta->box[m]._y));
            min_x = min(min_x, meta->box[m]._x);
            min_y = min(min_y, meta->box[m]._y);
            max_y = max(max_y, meta->box[m]._y);
        }
        const cv::Point *ppt[1] = {rook_points};
        int npt[] = {4};
        cv::polylines(image, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);

        // draw font
        int fontHeight = max(int(float(max_y - min_y) / 5.0f + 0.5f), 20);
        int thickness = -1;
        int linestyle = 8;
        int baseline = 0;
#ifdef OPENCV_HAS_FREETYPE
        cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData(fontFileName,0);

        cv::Size textSize=ft2->getTextSize(meta->label, fontHeight,thickness,&baseline);

        if (thickness>0) baseline+=thickness;

        // center the text
        cv::Point textOrg(min_x, min_y);

        ft2->putText(image, meta->label, textOrg, fontHeight,
                     cv::Scalar(255, 0,0), thickness, linestyle, true );
#endif
    }

    void Utility::ImageNet::imagenet_post_process(const std::vector<float>& output, const std::vector<int>& output_shape, std::vector<std::shared_ptr<ClsMeta>> &result, int topK){
        assert(1 == output_shape[0]); // only 1 batch
        std::vector<std::shared_ptr<ClsMeta>> tmp;
        tmp.resize(output_shape[1]);
        for(int i = 0; i < output_shape[1]; i++){
            std::shared_ptr<ClsMeta> meta = std::make_shared<ClsMeta>();
            meta->conf = output[i];
            meta->label = i;
            meta->label_str = _label_list[i];
            tmp[i] = meta;
        }
        std::sort(tmp.begin(), tmp.end(), [](const std::shared_ptr<ClsMeta> &a, const std::shared_ptr<ClsMeta> &b){ return a->conf > b->conf;});

        result.assign(tmp.begin(), tmp.begin() + topK);
    }
    void Utility::ImageNet::load_label(const std::string &path, std::vector<std::string>& m_vec){
        std::ifstream in(path);
        std::string line;

        if (in) {
            while (getline(in, line)) {
                m_vec.push_back(line);
            }
        } else {
            std::cout << "no such label file: " << path << ", exit the program..."
                      << std::endl;
            exit(1);
        }
    }

    void Utility::ImageNet::operator()(const vector<float> &output, const vector<int> &output_shape,
                                       vector<std::shared_ptr<ClsMeta>> &result, int topK) {
        imagenet_post_process(output, output_shape, result, topK);

    }
#endif
}


