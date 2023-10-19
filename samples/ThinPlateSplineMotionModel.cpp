//
// Created by TalkUHulk on 2023/10/3.
//


#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include "utility/face_align.h"
#if __linux__
#include <fstream>
#endif

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage, double cost) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r\b%3d%% %3.3fs [%.*s%*s]", val, cost, lpad, PBSTR, rpad, "");
    fflush(stdout);
}


int avd_network(AIDB::Interpreter* ins, void* data1, void* data2, std::vector<float> &kp_norm){

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    std::vector<void*> inputs{data1, data2};

    std::vector<std::vector<int>> input_shapes(2, {1, 50, 2});
    ins->forward(inputs, input_shapes, outputs, outputs_shape);

    kp_norm.assign(outputs[0].begin(), outputs[0].end());

    return 0;
}

int kp_detect(AIDB::Interpreter* ins, const cv::Mat& bgr, cv::Mat& blob, std::vector<float> &kp_source){
    blob = *ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

    kp_source.assign(outputs[0].begin(), outputs[0].end());

    return 0;
}

int face_detect(AIDB::Interpreter* ins, const cv::Mat& bgr, std::vector<std::shared_ptr<AIDB::FaceMeta>>& face_metas){
    cv::Mat blob = *ins << bgr;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

    assert(face_detect_input->scale_h() == face_detect_input->scale_w());
    AIDB::Utility::scrfd_post_process(outputs, face_metas, ins->width(), ins->height(), ins->scale_h());

    return 0;
}

int face_parsing(AIDB::Interpreter* ins, const cv::Mat& bgr, const std::shared_ptr<AIDB::FaceMeta>& face_meta, cv::Mat &mask){

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int>> outputs_shape;

    cv::Mat align;
    cv::Mat inverse_transformation_matrix;
    AIDB::faceAlign(bgr, align, face_meta, inverse_transformation_matrix, ins->width(), "ffhq");

    auto blob = *ins << align;

    cv::Mat src_image;
    *ins >> src_image;

    auto result = bgr.clone();

    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

    AIDB::Utility::bisenet_post_process(src_image, result, outputs[0], outputs_shape[0], false, {16, 18});

//    cv::imshow("result", result);
    mask.create(bgr.rows, bgr.cols, CV_32FC1);
    cv::Mat binary;
    cv::cvtColor(10*(255 - result), result, cv::COLOR_BGR2GRAY);
    cv::threshold(result, binary, 1, 255, cv::THRESH_BINARY);
    cv::warpAffine(binary, mask, inverse_transformation_matrix,
                   cv::Size(bgr.cols, bgr.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

//    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
//
//    cv::dilate(mask, mask, element);
    return 0;
}


int main(int argc, char** argv){

    if(argc != 6){
        std::cout << "xxx source_image driving_video backend[onnx,mnn,openvino] mode[0, 1, 2] save_path\n";
        return 0;
    }


    auto source_image = argv[1];
    auto driving_video = argv[2];
    auto backend = argv[3];
    auto mode = atoi(argv[4]);
    auto save_path = argv[5];
    auto raw = 1;//atoi(argv[6]); bad

    auto interpreter_kp = AIDB::Interpreter::createInstance("tpsmm_kp_detector", backend);
    AIDB::Interpreter* interpreter_avd = nullptr;
    if(mode == 2)
        interpreter_avd = AIDB::Interpreter::createInstance("tpsmm_avd_network", backend);

    auto interpreter_tpsmm = AIDB::Interpreter::createInstance("tpsmm", backend);

    if(nullptr == interpreter_kp || nullptr == interpreter_tpsmm){
        return -1;
    }

    auto interpreter_face_detect = AIDB::Interpreter::createInstance("scrfd_500m_kps", backend);

    auto source = cv::imread(source_image);

    std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas;
    face_detect(interpreter_face_detect, source, face_metas);

    cv::Mat mask;

    AIDB::Interpreter* interpreter_parsing = nullptr;
    if(!raw){
        interpreter_parsing = AIDB::Interpreter::createInstance("bisenet", backend);
        face_parsing(interpreter_parsing, source, face_metas[0], mask);
    }


    bool is_first = true;
    cv::Mat driving;
//    auto cap = cv::VideoCapture("/Users/hulk/Documents/CodeZoo/Thin-Plate-Spline-Motion-Model-main/assets/driving.mp4");
    auto cap = cv::VideoCapture(driving_video);


    if(!cap.isOpened()){
        return -1;
    }
    cap >> driving;
    if (driving.empty()) {
        return -1;
    }
    std::vector<std::shared_ptr<AIDB::FaceMeta>> driving_face_metas;
    face_detect(interpreter_face_detect, driving, driving_face_metas);

    std::shared_ptr<AIDB::FaceMeta> driving_face_meta_ex = std::make_shared<AIDB::FaceMeta>();;
    AIDB::Utility::Common::parse_roi_from_bbox(driving_face_metas[0], driving_face_meta_ex, driving.cols, driving.rows, 2.15, 0.14);

    cv::Mat driving_roi(driving, cv::Rect(driving_face_meta_ex->x1, driving_face_meta_ex->y1, driving_face_meta_ex->width(), driving_face_meta_ex->height()));

    for(int i = 0; i < driving_face_meta_ex->kps.size() / 2; i++){
        driving_face_meta_ex->kps[2 * i] -= driving_face_meta_ex->x1;
        driving_face_meta_ex->kps[2 * i + 1] -= driving_face_meta_ex->y1;
    }

    cv::Mat source_roi;
    cv::Mat inverse_transformation_matrix;
    if(raw){
        std::shared_ptr<AIDB::FaceMeta> source_meta_roi = std::make_shared<AIDB::FaceMeta>();
        AIDB::Utility::Common::parse_roi_from_bbox(face_metas[0], source_meta_roi, source.cols, source.rows, 2.15, 0.14);
        source_roi = cv::Mat(source, cv::Rect(source_meta_roi->x1, source_meta_roi->y1, source_meta_roi->width(), source_meta_roi->height()));
    } else{
        AIDB::faceAlign(source, driving_roi, source_roi, face_metas[0], driving_face_meta_ex, inverse_transformation_matrix, cv::Size(256, 256));
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    auto video_writer = cv::VideoWriter(save_path, codec, 25, raw == 1 ?cv::Size(256, 256): cv::Size(source.cols, source.rows));

    int frame_cnt = int(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // source kp
    cv::Mat blob_source;
    std::vector<float> kp_source;
    kp_detect(interpreter_kp, source_roi, blob_source, kp_source);

    std::vector<float> kp_driving_initial;

    std::vector<std::vector<int>> input_shapes = {{1, 50, 2}, {1, 50, 2}, {1, 3, 256, 256}, {1, 3, 256, 256}};

    int cnt = 0;
    while(true) {
        cap >> driving;
        if (driving.empty()) {
            break;
        }
        cnt ++;
        auto tic = std::chrono::high_resolution_clock::now();

        std::vector<float> kp_norm;
        std::vector<float> kp_driving;
        cv::Mat blob_driving;

        if(is_first){
            kp_detect(interpreter_kp, driving, blob_driving, kp_driving_initial);
            kp_driving = kp_driving_initial;
            is_first = false;
        } else{
            kp_detect(interpreter_kp, driving, blob_driving, kp_driving);
        }

        // kp norm
        switch (mode) {
            case 0:{
                kp_norm = kp_driving;
                break;
            }
            case 1:{
                AIDB::Utility::relative_kp(kp_source, kp_driving, kp_driving_initial, kp_norm);
                break;
            }
            case 2:{
                avd_network(interpreter_avd, kp_source.data(), kp_driving.data(), kp_norm);
                break;
            }
            default:{
                std::cout << "not support\n";
            }

        }

        //tpsmm
        std::vector<void*> inputs{kp_source.data(), kp_driving.data(), blob_source.data, blob_driving.data};
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<int>> outputs_shape;

        interpreter_tpsmm->forward(inputs, input_shapes, outputs, outputs_shape);

        int siz[] = {1, 3, 256, 256};

        cv::Mat blob_output(4, siz, CV_32F, outputs[0].data());
        std::vector<cv::Mat> generated;

        cv::dnn::imagesFromBlob(blob_output, generated);

        auto toc = std::chrono::high_resolution_clock::now();    //结束时间
        std::chrono::duration<double> elapsed = toc - tic;

        printProgress(cnt / double(frame_cnt), elapsed.count());
        cv::Mat generatedRGB;
        generated[0].convertTo(generatedRGB, CV_8U,  255);
        cv::cvtColor(generatedRGB, generatedRGB, cv::COLOR_BGR2RGB);


        if(raw){
            video_writer << generatedRGB;
//            cv::imshow("source_generated", generatedRGB);
//            cv::waitKey(0);
        } else {

            cv::Mat source_generated;
            source_generated.create(source.rows, source.cols, CV_32FC3);

            cv::warpAffine(generatedRGB, source_generated, inverse_transformation_matrix,
                           cv::Size(source.cols, source.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            source_generated.convertTo(source_generated, CV_8UC3);

            std::vector<std::shared_ptr<AIDB::FaceMeta>> face_metas2;
            face_detect(interpreter_face_detect, source_generated, face_metas2);

            cv::Mat blur;
            GaussianBlur(source_generated, blur, cv::Size(3, 3), 15);
            addWeighted(source_generated, 1.5, blur, -0.5, 0, source_generated);

            cv::Mat mask2;
            face_parsing(interpreter_parsing, source_generated, face_metas2[0], mask2);
//        face_parsing2(interpreter_3ddfa, source_generated, face_metas2[0], mask2);

            cv::bitwise_or(mask, mask2, mask2);

            auto black = source.clone();
            black = 0;
            auto source_tmp = source.clone();
            source_tmp = 0;

            source.copyTo(source_tmp, 255 - mask2);

            cv::bitwise_and(source_generated, black, source_generated, 255 - mask2);

            source_generated += source_tmp;

            video_writer << source_generated;
//            std::cout << source_generated.size << std::endl;
//            cv::imshow("source_generated", source_generated);
//            cv::waitKey(1);

        }


    }

    AIDB::Interpreter::releaseInstance(interpreter_kp);
    AIDB::Interpreter::releaseInstance(interpreter_tpsmm);
    AIDB::Interpreter::releaseInstance(interpreter_face_detect);
    if(interpreter_parsing)
        AIDB::Interpreter::releaseInstance(interpreter_parsing);
    if(interpreter_avd)
        AIDB::Interpreter::releaseInstance(interpreter_avd);

    video_writer.release();
    cap.release();
    return 0;
}