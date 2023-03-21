//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include <memory>
#if __linux__
#include <fstream>
#endif


void plot_pose_cube(cv::Mat& image, const std::vector<float> &pose, float tdx, float tdy, float size){
    auto p = pose[0] * CV_PI / 180;
    auto y = -(pose[1] * CV_PI / 180);
    auto r = pose[2] * CV_PI / 180;

    auto face_x = tdx - 0.50 * size;
    auto face_y = tdy - 0.50 * size;

    auto x1 = size * (cos(y) * cos(r)) + face_x;
    auto y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y;
    auto x2 = size * (-cos(y) * sin(r)) + face_x;
    auto y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y;
    auto x3 = size * (sin(y)) + face_x;
    auto y3 = size * (-cos(y) * sin(p)) + face_y;

    // Draw base in red
    cv::line(image, cv::Point(int(face_x), int(face_y)), cv::Point(int(x1), int(y1)), cv::Scalar(0, 0, 255), 3);
    cv::line(image, cv::Point(int(face_x), int(face_y)), cv::Point(int(x2), int(y2)), cv::Scalar(0, 0, 255), 3);
    cv::line(image, cv::Point(int(x2), int(y2)), cv::Point(int(x2 + x1 - face_x), int(y2 + y1 - face_y)), cv::Scalar(0, 0, 255), 3);
    cv::line(image, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + x2 - face_x), int(y1 + y2 - face_y)), cv::Scalar(0, 0, 255), 3);
    // Draw pillars in blue
    cv::line(image, cv::Point(int(face_x), int(face_y)), cv::Point(int(x3), int(y3)), cv::Scalar(255, 0, 0), 2);
    cv::line(image, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + x3 - face_x), int(y1 + y3 - face_y)), cv::Scalar(255, 0, 0), 2);
    cv::line(image, cv::Point(int(x2), int(y2)), cv::Point(int(x2 + x3 - face_x), int(y2 + y3 - face_y)), cv::Scalar(255, 0, 0), 2);
    cv::line(image, cv::Point(int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(255, 0, 0), 2);
    // Draw top in green
    cv::line(image, cv::Point(int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(image, cv::Point(int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(image, cv::Point(int(x3), int(y3)), cv::Point(int(x3 + x1 - face_x), int(y3 + y1 - face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(image, cv::Point(int(x3), int(y3)), cv::Point(int(x3 + x2 - face_x), int(y3 + y2 - face_y)), cv::Scalar(0, 255, 0), 2);

}

void test_landmark(AIDB::Interpreter* det_ins, AIDB::Interpreter* ldm_ins, const cv::Mat& bgr, cv::Mat &result){
    cv::Mat blob = *det_ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    det_ins->forward((float*)blob.data, det_ins->width(), det_ins->height(), det_ins->channel(), outputs, outputs_shape);

    std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;

    assert(face_detect_input->scale_h() == face_detect_input->scale_w());
    AIDB::Utility::scrfd_post_process(outputs, face_metas, det_ins->width(), det_ins->height(), det_ins->scale_h());
    for(auto &face_meta: face_metas){

        outputs.clear();
        outputs_shape.clear();
        std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
        AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, bgr.cols, bgr.rows, 1.28, 0.14);
        cv::Mat roi(bgr, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));
        blob = *ldm_ins << roi;
        ldm_ins->forward((float*)blob.data, ldm_ins->width(), ldm_ins->height(), ldm_ins->channel(), outputs, outputs_shape);

        AIDB::Utility::pfpld_post_process(outputs, face_meta_roi, face_meta, 98);
    }

    result = bgr.clone();

    for(auto &face_meta: face_metas){
        int baseLine = 0;
        char text[10];
        sprintf(text, "%.2f", face_meta->score);

        cv::Size label_size = cv::getTextSize(text,
                                              cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int y = face_meta->y1 - label_size.height;
        int x = face_meta->x1;

        cv::rectangle(result, cv::Rect(cv::Point(x, y - baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(result, text, cv::Point(x, y + label_size.height - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255));

        cv::rectangle(result, cv::Point(int(face_meta->x1), int(face_meta->y1)),
                      cv::Point(int(face_meta->x2), int(face_meta->y2)),
                      cv::Scalar(255, 0, 0), 2);

        for(int n = 0; n < face_meta->kps.size(); n++){
            cv::circle(result, cv::Point(int(face_meta->kps[2 * n]), int(face_meta->kps[2 * n + 1])), 3, cv::Scalar(0, 0, 255), -1);
        }

        plot_pose_cube(result, face_meta->pose, face_meta->kps[2 * 54], face_meta->kps[2 * 54 + 1], face_meta->width());

    }
}

int main(int argc, char** argv){

    auto face_detect_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == face_detect_interpreter){
        return -1;
    }

    auto face_landmark_interpreter = AIDB::Interpreter::createInstance(argv[3], argv[4]);

    if(nullptr == face_landmark_interpreter){
        return -1;
    }

    int input_type = std::atoi(argv[5]);
    std::string input_file = argv[6];


    if(0 == input_type){
        auto bgr = cv::imread(input_file);
        cv::Mat result;
        test_landmark(face_detect_interpreter, face_landmark_interpreter, bgr, result);
        cv::imwrite("FaceLandMark.jpg", result);
#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("FaceLandMark", result);
            cv::waitKey();
        }
#else
        cv::imshow("FaceLandMark", result);
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

//        writer.open("face_ladmark.mp4", codec, 25, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        cv::Mat bgr;
        cv::Mat result;
        while (cap.read(bgr)) {
            test_landmark(face_detect_interpreter, face_landmark_interpreter, bgr, result);
//            writer << result;
            cv::imshow("Face", result);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cap.release();
//        writer.release();

    }


    AIDB::Interpreter::releaseInstance(face_detect_interpreter);
    AIDB::Interpreter::releaseInstance(face_landmark_interpreter);

    return 0;
}