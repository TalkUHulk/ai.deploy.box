//
// Created by TalkUHulk on 2022/10/25.
//

#include "backend/ncnn/NCNNParameter.hpp"
#include <string>
namespace AIDB {


    NCNNParameter::NCNNParameter(const YAML::Node& node): Parameter(node){

        _model_name = node["name"].as<std::string>();
        _model_path = node["model"].as<std::string>() + ".bin";
        _param_path = node["model"].as<std::string>() + ".param";


        auto detail = node["detail"];

        for(auto in: detail["input_nodes"]){

            if(in["format"].IsDefined() && in["format"].as<std::string>() == "VECTOR") {
                std::vector<int> cur_shape;
                if(in["shape"]["batch"].IsDefined()) cur_shape.push_back(in["shape"]["batch"].as<int>());
                if(in["shape"]["channel"].IsDefined()) cur_shape.push_back(in["shape"]["channel"].as<int>());
                if(in["shape"]["height"].IsDefined()) cur_shape.push_back(in["shape"]["height"].as<int>());
                if(in["shape"]["width"].IsDefined()) cur_shape.push_back(in["shape"]["width"].as<int>());

                _input_nodes.insert({in["input_name"].as<std::string>(), cur_shape});

            }else if(in["format"].IsDefined() && in["format"].as<std::string>() == "NHWC"){
                _input_nodes.insert({in["input_name"].as<std::string>(),
                                     {in["shape"]["batch"].as<int>(), in["shape"]["height"].as<int>(), in["shape"]["width"].as<int>() == -1?-2:in["shape"]["width"].as<int>(), in["shape"]["channel"].as<int>()}
                                    });

            } else{
                _input_nodes.insert({in["input_name"].as<std::string>(),
                                     {in["shape"]["batch"].as<int>(), in["shape"]["channel"].as<int>(), in["shape"]["height"].as<int>(), in["shape"]["width"].as<int>() == -1?-2:in["shape"]["width"].as<int>()}
                                    });
            }

        }
        for(auto on: detail["output_nodes"]){
            _output_node_name.push_back(on.as<std::string>());
        }

        if(detail["register_layer"].IsDefined()){
            _register_layers.push_back(detail["register_layer"].as<std::string>());
        }
//        _mean = {node["means"][0].as<float>(), node["means"][1].as<float>(), node["means"][2].as<float>()};
//        _norm = {1.0f / node["norms"][0].as<float>(), 1.0f / node["norms"][1].as<float>(), 1.0f / node["norms"][2].as<float>()};

        _numThread = detail["num_thread"].as<int>();
        _encrypt = detail["encrypt"].as<bool>();
    }

    NCNNParameter::NCNNParameter(const std::string &config): Parameter(config){

        auto node = YAML::Load(config);
        _model_name = node["name"].as<std::string>();

        for(auto in: node["input_nodes"]){

            if(in["format"].IsDefined() && in["format"].as<std::string>() == "VECTOR") {
                std::vector<int> cur_shape;
                if(in["shape"]["batch"].IsDefined()) cur_shape.push_back(in["shape"]["batch"].as<int>());
                if(in["shape"]["channel"].IsDefined()) cur_shape.push_back(in["shape"]["channel"].as<int>());
                if(in["shape"]["height"].IsDefined()) cur_shape.push_back(in["shape"]["height"].as<int>());
                if(in["shape"]["width"].IsDefined()) cur_shape.push_back(in["shape"]["width"].as<int>());

                _input_nodes.insert({in["input_name"].as<std::string>(), cur_shape});

            }else if(in["format"].IsDefined() && in["format"].as<std::string>() == "NHWC"){
                _input_nodes.insert({in["input_name"].as<std::string>(),
                                     {in["shape"]["batch"].as<int>(), in["shape"]["height"].as<int>(), in["shape"]["width"].as<int>() == -1?-2:in["shape"]["width"].as<int>(), in["shape"]["channel"].as<int>()}
                                    });

            } else{
                _input_nodes.insert({in["input_name"].as<std::string>(),
                                     {in["shape"]["batch"].as<int>(), in["shape"]["channel"].as<int>(), in["shape"]["height"].as<int>(), in["shape"]["width"].as<int>() == -1?-2:in["shape"]["width"].as<int>()}
                                    });
            }

        }
        for(auto on: node["output_nodes"]){
            _output_node_name.push_back(on.as<std::string>());
        }

        if(node["register_layer"].IsDefined()){
            _register_layers.push_back(node["register_layer"].as<std::string>());
        }
//        _mean = {node["means"][0].as<float>(), node["means"][1].as<float>(), node["means"][2].as<float>()};
//        _norm = {1.0f / node["norms"][0].as<float>(), 1.0f / node["norms"][1].as<float>(), 1.0f / node["norms"][2].as<float>()};

        _numThread = node["num_thread"].as<int>();
    }

}