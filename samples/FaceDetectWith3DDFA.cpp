//
// Created by TalkUHulk on 2023/2/5.
//


#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include "td_obj.h"
#include <memory>
#if __linux__
#include <fstream>
#endif

void test_tddfav2(AIDB::Interpreter* det_ins, AIDB::Interpreter* tdd_ins, const cv::Mat& bgr, cv::Mat &result, bool dense=false, const std::string &obj_file="extra/spiderman2_obj.obj"){
    cv::Mat blob = *det_ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    det_ins->forward((float*)blob.data, det_ins->width(), det_ins->height(), det_ins->channel(), outputs, outputs_shape);


    std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;

    assert(det_ins->scale_h() == det_ins->scale_w());
    AIDB::Utility::scrfd_post_process(outputs, face_metas, det_ins->width(), det_ins->height(), det_ins->scale_h());

    result = bgr.clone();
    for(auto &face_meta: face_metas){

        std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
        AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, bgr.cols, bgr.rows, 1.58, 0.14);
        cv::Mat roi(bgr, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));

        blob = *tdd_ins << roi;
        outputs.clear();
        outputs_shape.clear();
        tdd_ins->forward((float*)blob.data, tdd_ins->width(), tdd_ins->height(), tdd_ins->channel(), outputs, outputs_shape);
        std::vector<float> vertices, pose, sRt;

        AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

        if(dense){
            if(obj_file.empty()){
                for(int i = 0; i < vertices.size() / 3; i+=6){
                    cv::circle(result, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(255, 255, 255), -1);
                }
            } else{
//                AIDB::Utility::TddfaUtility::tddfa_rasterize(result, vertices, obj_file.c_str(), 0);
//                AIDB::Utility::TddfaUtility::tddfa_rasterize(result, vertices, spider_man_obj, 1);
                AIDB::Utility::TddfaUtility::tddfa_rasterize(result, vertices, hulk_obj, 1, false);

            }
        }
        else{
            for(int i = 0; i < vertices.size() / 3; i++){
                cv::circle(result, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(255, 255, 255), -1);
            }
            AIDB::Utility::TddfaUtility::plot_pose_box(result, sRt, vertices, 68);
        }

    }

}

int main(int argc, char** argv){

    auto face_detect_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == face_detect_interpreter){
        return -1;
    }

    auto face_3ddfa_interpreter = AIDB::Interpreter::createInstance(argv[3], argv[4]);

    if(nullptr == face_3ddfa_interpreter){
        return -1;
    }

    auto idx = std::string(argv[3]).find("dense");
    bool dense = true;
    if(idx == std::string::npos){
        dense = false;
    }

    int input_type = std::atoi(argv[5]);
    std::string input_file = argv[6];


    if(0 == input_type){
        auto bgr = cv::imread(input_file);
        cv::Mat result;
        test_tddfav2(face_detect_interpreter, face_3ddfa_interpreter, bgr, result, dense);
        cv::imwrite("Face3DDFAV2.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("3DDFAV2", result);
            cv::waitKey();
        }
#else
        cv::imshow("3DDFAV2", result);
        cv::waitKey();
#endif

    } else {
        cv::VideoCapture cap;
//        cv::VideoWriter writer;
//        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if(input_file.size() < 2)
            cap.open(std::atoi(input_file.c_str()));
        else
            cap.open(input_file);

//        writer.open("face_3ddfav2.mp4", codec, 25, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            test_tddfav2(face_detect_interpreter, face_3ddfa_interpreter, bgr, result, dense);
//            writer << result;
            cv::imshow("3DDFAV2", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
//        writer.release();
    }


    AIDB::Interpreter::releaseInstance(face_detect_interpreter);
    AIDB::Interpreter::releaseInstance(face_3ddfa_interpreter);

    return 0;
}