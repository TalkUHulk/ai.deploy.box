//
// Created by TalkUHulk on 2022/10/18.
//

#ifndef AIENGINE_PARAMETER_HPP
#define AIENGINE_PARAMETER_HPP

#include "AIDBDefine.h"
#include <iostream>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>
#include <string>

namespace AIDB {

    class AIDB_PUBLIC Parameter {
    public:
        Parameter() = delete;
        virtual ~Parameter() = default;
        explicit Parameter(const YAML::Node& ){};
        explicit Parameter(const std::string& ){};

    public:
        int _numThread=4;
//        std::vector<float> _mean{0.0f, 0.0f, 0.0f, 0.0f};    /*!< 均值*/
//        std::vector<float> _norm {1.0f, 1.0f, 1.0f, 1.0f};    /*!< 方差*/
        bool _encrypt{};      /*!< 模型是否加密*/
        EngineID _engine_id{};      /*!< 模型对应ID*/
        Device _device=CPU;
        bool _dynamic=false;
        std::string _model_path{};
        std::string _param_path{};
        // To Do
        // 有些模型可能有多个输入输出节点，后续再修改
        std::map<std::string, std::vector<int>> _input_nodes;  /*!< 输入节点信息 {name: shape}*/
        std::vector<std::string> _input_node_name{};  /*!< 输入节点名称*/
        std::vector<std::string> _output_node_name{};  /*!< 输出节点名称*/
        std::string _model_name{};  /*!< 模型名称*/
        std::string _backend_name{};  /*!< 模型名称*/
        std::vector<std::string> _register_layers{}  /*!< 自定义layer*/;
    };
}

#endif //AIENGINE_PARAMETER_HPP
