//
// Created by TalkUHulk on 2022/10/17.
//

#ifndef AIENGINE_ENGINE_HPP
#define AIENGINE_ENGINE_HPP

#include "AIDBDefine.h"
#include <string>
#include "Parameter.hpp"
#include "StatusCode.h"
#include "utility/log.h"
#include "utility/Utility.h"

namespace AIDB {
    class AIDB_PUBLIC Engine {
    public:
        Engine() = default;
        virtual StatusCode init(const Parameter&) = 0;
        virtual StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) = 0;
        virtual ~Engine(){};
        virtual void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) = 0;
//        virtual void forward(const float *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) = 0;
        std::vector<std::string> _output_node_name;
        std::map<std::string, std::vector<int>> _input_nodes;  /*!< 输入节点信息*/
        bool _dynamic=false;
        std::string _model_name = "default";
        std::string _backend_name = "default";

    };
}

#endif //ENGINE_HPP
