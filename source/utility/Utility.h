//
// Created by TalkUHulk on 2023/2/5.
//

#ifndef AIDEPLOYBOX_UTILITY_H
#define AIDEPLOYBOX_UTILITY_H

#include "AIDBData.h"
#include "AIDBDefine.h"
#include <iostream>

#include <memory>
#include <yaml-cpp/yaml.h>
#ifdef ENABLE_NCNN_WASM
#include <simpleocv.h>
#include "Eigen/Dense"
#else
#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
#endif


namespace AIDB{

#define ONE_MB 104857600
#define AIDB_TRACE "AIDB_TRACE"
#define AIDB_DEBUG "AIDB_DEBUG"
#define AIDB_INFO "AIDB_INFO"
#define AIDB_WARNING "AIDB_WARNING"
#define AIDB_ERROR "AIDB_ERROR"
#define AIDB_CRITICAL "AIDB_CRITICAL"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)


    struct GridAndStride{
        int grid0;
        int grid1;
        int stride;
    };

    static cv::Scalar parsing_part_colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
                                          cv::Scalar(255, 0, 85), cv::Scalar(255, 0, 170),
                                          cv::Scalar(0, 255, 0), cv::Scalar(85, 255, 0), cv::Scalar(170, 255, 0),
                                          cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
                                          cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
                                          cv::Scalar(0, 85, 255), cv::Scalar(0, 170, 255),
                                          cv::Scalar(255, 255, 0), cv::Scalar(255, 255, 85), cv::Scalar(255, 255, 170),
                                          cv::Scalar(255, 0, 255), cv::Scalar(255, 85, 255), cv::Scalar(255, 170, 255),
                                          cv::Scalar(0, 255, 255), cv::Scalar(85, 255, 255), cv::Scalar(170, 255, 255)};

    template<class T>
    inline T clip(float x, T _min, T _max){
        return (T)fmax(fmin(x, _max), _min);
    }


    class AIDB_PUBLIC Utility {
    public:

        class Common{
        public:
            template<class T>
            static float IOU(const std::shared_ptr<T> meta1, const std::shared_ptr<T> meta2);
            template<class T>
            static void NMS(std::vector<std::shared_ptr<T>> &metas, std::vector<std::shared_ptr<T>> &keep_metas, float threshold);

            static void parse_roi_from_bbox(std::shared_ptr<FaceMeta> org_meta,  std::shared_ptr<FaceMeta> dst_meta, int image_w, int image_h, float expand_ratio=1.0f, float re_centralize_ratio=0.0f);

            static void draw_objects(const cv::Mat& src, cv::Mat &dst, const std::vector<std::shared_ptr<ObjectMeta>>& objects);
        };

        static void mobile_sam_post_process(const std::vector<float> &output,
                                            const cv::Mat &source,
                                            cv::Mat &result,
                                            float scale,
                                            const cv::Scalar &color,
                                            float alpha=0.5);
        static void scrfd_post_process(const std::vector<std::vector<float>> &outputs, std::vector<std::shared_ptr<FaceMeta>> &face_metas, int target_w, int target_h, float det_scale, float nms_thresh=0.4);

        static void pfpld_post_process(const std::vector<std::vector<float>> &outputs, const std::shared_ptr<FaceMeta> scale_face_metas, std::shared_ptr<FaceMeta> dst_face_metas, int pts_num=98);

        static void movenet_post_process(const cv::Mat &src_image, cv::Mat &result_image,
                                         const std::vector<std::vector<float>> &outputs,
                                         const std::vector<std::vector<int>> &outputs_shape);

        static void movenet_post_process(const cv::Mat &src_image,
                                         const std::vector<std::vector<float>> &outputs,
                                         const std::vector<std::vector<int>> &outputs_shape,
                                         std::vector<std::vector<float>>& decoded_keypoints);

        static void tddfa_post_process(const std::vector<std::vector<float>> &outputs, const std::vector<std::vector<int>> &outputs_shape,
                           const std::shared_ptr<FaceMeta>& face_meta, std::vector<float> &vertices, std::vector<float> &pose, std::vector<float> &sRt, int target=120);
#ifndef ENABLE_NCNN_WASM
        static void animated_gan_post_process(const std::vector<float> &output,
                                                const std::vector<int> &outputs_shape, cv::Mat &animated);

        static void bisenet_post_process(const cv::Mat &src_image, cv::Mat &parsing_image, const std::vector<float> &outputs,
                                           const std::vector<int> &outputs_shape);

        static void stylegan_post_process(cv::Mat &result_image,
                                         const std::vector<float> &output,
                                         const std::vector<int> &output_shape);
#endif
        // without grid
        static void yolov7_post_process(const std::vector<float> &output,
                                        const std::vector<int> &output_shape,
                                        std::vector<std::shared_ptr<ObjectMeta>> &results,
                                        float conf_thresh, float nms_thresh,
                                        float scale);
        // with grid
        static void yolov7_post_process(const std::vector<std::vector<float>> &outputs,
                                        const std::vector<std::vector<int>> &outputs_shape,
                                        std::vector<std::shared_ptr<ObjectMeta>> &results,
                                        float conf_thresh, float nms_thresh,
                                        float scale);

        static void yolov8_post_process(const std::vector<float> &output,
                                        const std::vector<int> &output_shape,
                                        std::vector<std::shared_ptr<ObjectMeta>> &results,
                                        float conf_thresh, float nms_thresh,
                                        float scale);

        class MoveNetUtility{
        public:
            static int line_map[20][2];

            // only support 1 batch
            static void MoveNetDecode(const std::vector<std::vector<float>> &outputs,
                                      const std::vector<std::vector<int>> &outputs_shape,
                                      std::vector<std::vector<float>> &decoded_keypoints,
                                      int joints_num=17, float heatmap_thresh=0.1, int target_size=48);
            static void get_max_points(const std::vector<float> &output, int height, int width, int channel, float &max_x, float& max_y, bool center=true);
        };

        class TddfaUtility{
        public:
            static void similar_transform(const std::vector<float> &input,
                                          int bs, int pts_num,
                                          std::vector<float> &output,
                                          const std::shared_ptr<FaceMeta>& facemeta, int target=120);
            static void matrix2angle(const std::vector<float> &sRt, std::vector<float> &pose);
            static void calc_pose(const std::vector<float> &input, std::vector<float> &sRt, std::vector<float> &pose);
            static void calc_hypotenuse_and_mean(const std::vector<float> &ver, float &mean_x, float &mean_y, float &llength, int kpts_num=68, int mean_index_end=27);
            static Eigen::Matrix<float, 10, 4, Eigen::RowMajor> build_camera_box(float rear_size=90);
#ifndef ENABLE_NCNN_WASM
            static void plot_pose_box(cv::Mat &img, const std::vector<float> &P, const std::vector<float> &ver, int kpts_num, cv::Scalar color=cv::Scalar(40, 255, 0), int line_width=2);
#endif
            static void load_obj(const char *obj_fp, std::vector<float> &vertices, std::vector<float> &colors, std::vector<int> &triangles, int nver, int ntri);

            // color_rgb_swap: obj color 是RGB,如果image为BGR需要swap
            static void tddfa_rasterize(cv::Mat &image, const std::vector<float> &vertices, const char *obj, int obj_type, bool color_rgb_swap=true);

            static void load_obj2(const char *obj, std::vector<float> &vertices, std::vector<float> &colors, std::vector<int> &triangles, int nver, int ntri);

        };

#ifndef ENABLE_NCNN_WASM
        class ImageNet{
        public:
            ImageNet(const std::string &label_path){
                load_label(label_path, _label_list);
            };
            void operator()(const std::vector<float>& output, const std::vector<int>& output_shape, std::vector<std::shared_ptr<ClsMeta>> &result, int topK=3);
            void imagenet_post_process(const std::vector<float>& output, const std::vector<int>& output_shape, std::vector<std::shared_ptr<ClsMeta>> &result, int topK=3);
            static void load_label(const std::string &path, std::vector<std::string>& m_vec);
        private:
            std::vector<std::string> _label_list;
        };


#endif

        class YoloX{
        public:
            YoloX(int target_size=640, float conf_thresh=0.25, float nms_thresh=0.45, const std::vector<int>& strides={8, 16, 32}):_target_size(target_size), _nms_thresh(nms_thresh), _conf_thresh(conf_thresh){
                generate_grids_and_stride(target_size, strides);
            }
            void generate_grids_and_stride(int target_size, const std::vector<int>& strides);
            void generate_proposals(const std::vector<float>& feat_blob, const std::vector<int>& outputs_shape, float prob_threshold, std::vector<std::shared_ptr<ObjectMeta>>& objects);
            void forward(const std::vector<float>& outputs, const std::vector<int>& outputs_shape, std::vector<std::shared_ptr<ObjectMeta>> &result, int src_width, int src_height, float scale);
            void operator()(const std::vector<float>& outputs, const std::vector<int>& outputs_shape, std::vector<std::shared_ptr<ObjectMeta>> &result, int src_width, int src_height, float scale);
        private:
            std::vector<GridAndStride> _grid_strides;
            int _target_size;
            float _nms_thresh;
            float _conf_thresh;
        };

#ifndef ENABLE_NCNN_WASM
        class PPOCR{
        public:
            PPOCR(float det_db_thresh=0.3, float det_db_box_thresh=0.5,
                  float det_db_unclip_ratio=2.0, bool use_dilation=false,
                  const std::string &det_db_score_mode="slow", float cls_thresh=0.9, const std::string &crnn_label_path="extra/ppocr_keys_v1.txt"):
            _det_db_thresh(det_db_thresh), _det_db_box_thresh(det_db_box_thresh), _det_db_unclip_ratio(det_db_unclip_ratio),
            _use_dilation(use_dilation), _det_db_score_mode(det_db_score_mode), _cls_thresh(cls_thresh){

                AIDB::Utility::PPOCR::ReadDict(crnn_label_path, _label_list);
                _label_list.insert(_label_list.begin(),
                                  "#"); // blank char for ctc
                _label_list.emplace_back(" ");
            };

            void dbnet_post_process(const std::vector<float>& output, const std::vector<int>& output_shape,
                                    std::vector<std::shared_ptr<OcrMeta>> &det_results,
                                    float ratio_h, float ratio_w, const cv::Mat& srcimg);
            void cls_post_process(const std::vector<float>& output, const std::vector<int>& output_shape,
                                         const cv::Mat& src, cv::Mat& dst, std::shared_ptr<AIDB::OcrMeta> ocr_result);
            void crnn_post_process(const std::vector<float>& output, const std::vector<int>& output_shape, std::shared_ptr<AIDB::OcrMeta> ocr_result);
            static void GetRotateCropImage(const cv::Mat &src, cv::Mat &dst, const std::shared_ptr<OcrMeta> &meta);
            static void ReadDict(const std::string &path, std::vector<std::string>& m_vec);
            static void draw_objects(cv::Mat &image, const std::shared_ptr<OcrMeta> &meta, const std::string fontFileName="extra/simkai.ttf");
        private:

            void BoxesFromBitmap(
                    const cv::Mat& pred, const cv::Mat& bitmap,
                    std::vector<std::shared_ptr<OcrMeta>> &boxes);
            static void FilterTagDetRes(std::vector<std::shared_ptr<OcrMeta>> boxes, float ratio_h,
                    float ratio_w, const cv::Mat& srcimg, std::vector<std::shared_ptr<OcrMeta>> &det_results);
            std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);
            std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid);
            static float PolygonScoreAcc(const std::vector<cv::Point> &contour, const cv::Mat& pred);
            static float BoxScoreFast(const std::vector<std::vector<float>>& box_array, const cv::Mat& pred);
            void UnClip(const std::vector<std::vector<float>> &box, float unclip_ratio, cv::RotatedRect &rotate_rect);
            static void GetContourArea(const std::vector<std::vector<float>> &box, float unclip_ratio, float &distance);
            static std::shared_ptr<OcrMeta> OrderPointsClockwise(const std::shared_ptr<OcrMeta> &pts);

        private:
            std::vector<std::string> _label_list;
            float _cls_thresh = 0.9;
            float _det_db_thresh=0.3;
            float _det_db_box_thresh=0.5;
            float _det_db_unclip_ratio=2.0;
            bool _use_dilation=false;
            int _min_size = 3;
            int _max_candidates = 1000;
            std::string _det_db_score_mode = "slow";
        };
#endif
    };
}



#endif //AIDEPLOYBOX_UTILITY_H
