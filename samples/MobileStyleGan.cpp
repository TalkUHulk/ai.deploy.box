//
// Created by TalkUHulk on 2023/2/5.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Interpreter.h"
#include "utility/Utility.h"
#include <chrono>
#if __linux__
#include <fstream>
#endif


int main(int argc, char** argv){

    auto mapping_interpreter = AIDB::Interpreter::createInstance(argv[1], argv[2]);

    if(nullptr == mapping_interpreter){
        return -1;
    }

    auto syn_interpreter = AIDB::Interpreter::createInstance(argv[3], argv[4]);

    if(nullptr == syn_interpreter){
        return -1;
    }

    std::vector<float> z(512);

    while(true){
        AIDB::Utility::randn(z.data(), z.size());
        std::vector<std::vector<float>> outputs_map;
        std::vector<std::vector<int>> outputs_shape_map;

        mapping_interpreter->forward((float*)z.data(), 512, 1, 1, outputs_map, outputs_shape_map);

        std::vector<std::vector<float>> outputs_syn;
        std::vector<std::vector<int>> outputs_shape_syn;

        syn_interpreter->forward((float*)outputs_map[0].data(), 512, 1, 1, outputs_syn, outputs_shape_syn);

        cv::Mat generated;
        AIDB::Utility::stylegan_post_process(generated, outputs_syn[0], outputs_shape_syn[0]);

        cv::Mat Z(256, 256, CV_32FC1, z.data());


#if __linux__
        // docker
        std::ifstream f("/.dockerenv");
        if(!f.good()){
            cv::imshow("noise", Z);
            cv::imshow("generated", generated);
        } else {
             cv::imwrite("noise.jpg", Z);
              cv::imwrite("MobileStyleGan.jpg", generated);
        }
#else
        cv::imshow("noise", Z);
        cv::imshow("MobileStyleGan", generated);
#endif

        if (cv::waitKey(100) == 27) {
            break;
        }
    }
    return 0;
}