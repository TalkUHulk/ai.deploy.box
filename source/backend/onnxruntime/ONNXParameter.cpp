//
// Created by TalkUHulk on 2022/10/18.
//

#include "backend/onnxruntime/ONNXParameter.hpp"
#include "core/Common.hpp"
namespace AIDB {

    ONNXParameter::ONNXParameter(const YAML::Node& node): Parameter(node){
        _backend_name = "onnx";
        _model_name = node["name"].as<std::string>();
        _model_path = node["model"].as<std::string>() + ".onnx";

        if(node["lora"].IsDefined()){
            std::cout << "这里" <<node["lora"] << std::endl;
            for(auto lora: node["lora"]){
                _lora_path.emplace_back(lora.as<std::string>());
            }
        }
        auto detail = node["detail"];
        for(auto in: detail["input_nodes"]){

            auto format = in["format"].as<std::string>();
            std::vector<int> cur_shape;
            for(const auto fm: format){
                if('N' == fm)
                    cur_shape.push_back(in["shape"]["batch"].as<int>());
                if('C' == fm)
                    cur_shape.push_back(in["shape"]["channel"].as<int>());
                if('H' == fm)
                    cur_shape.push_back(in["shape"]["height"].as<int>() < 0 ? -1:in["shape"]["height"].as<int>());
                if('W' == fm)
                    cur_shape.push_back(in["shape"]["width"].as<int>() < 0 ? -2: in["shape"]["width"].as<int>());

            }
            _input_node_name.emplace_back(in["input_name"].as<std::string>());
            _input_nodes.insert({in["input_name"].as<std::string>(), cur_shape});
            if(in["type"].IsDefined())
                _input_types.insert({in["input_name"].as<std::string>(), in["type"].as<std::string>()});
            else
                _input_types.insert({in["input_name"].as<std::string>(), "f32"});

//            if(in["format"].IsDefined() && in["format"].as<std::string>() == "VECTOR") {
//                std::vector<int> cur_shape;
//                if(in["shape"]["batch"].IsDefined()) cur_shape.push_back(in["shape"]["batch"].as<int>());
//                if(in["shape"]["channel"].IsDefined()) cur_shape.push_back(in["shape"]["channel"].as<int>());
//                if(in["shape"]["height"].IsDefined()) cur_shape.push_back(in["shape"]["height"].as<int>());
//                if(in["shape"]["width"].IsDefined()) cur_shape.push_back(in["shape"]["width"].as<int>());
//
//                _input_nodes.insert({in["input_name"].as<std::string>(), cur_shape});
//
//            }else if(in["format"].IsDefined() && in["format"].as<std::string>() == "NHWC"){
//                _input_nodes.insert({in["input_name"].as<std::string>(),
//                                     {in["shape"]["batch"].as<int>(), in["shape"]["height"].as<int>(), in["shape"]["width"].as<int>() == -1?-2:in["shape"]["width"].as<int>(), in["shape"]["channel"].as<int>()}
//                                    });
//
//            } else if(in["format"].IsDefined() && in["format"].as<std::string>() == "NCHW"){
//                _input_nodes.insert({in["input_name"].as<std::string>(),
//                                     {in["shape"]["batch"].as<int>(), in["shape"]["channel"].as<int>(), in["shape"]["height"].as<int>(), in["shape"]["width"].as<int>() == -1?-2:in["shape"]["width"].as<int>()}
//                                    });
//            } else{
//                //TO DO
//            }

        }

        for(auto on: detail["output_nodes"]){
            _output_node_name.push_back(on.as<std::string>());
        }

        _numThread = detail["num_thread"].as<int>();
        _encrypt = detail["encrypt"].as<bool>();
        _device = deviceType(detail["device"].as<std::string>());
    }

}