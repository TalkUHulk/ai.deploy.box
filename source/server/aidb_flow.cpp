//
// Created by TalkUHulk on 2023/6/29.
//

#include "aidb_flow.hpp"
#include "reflect.hpp"
#include "utility/log.h"

namespace AiDBServer {
    AiDBFlow::AiDBFlow(const json &input) {
        auto model = input["model"];
        auto backend = input["backend"];
        auto zoo = input["zoo"];
        if (backend.size() != model.size() && backend.size() != 1) {
            spdlog::get(AIDB_DEBUG)->error("Input Parameter Invalid, backend size != model size");
            return;
        }
        while (backend.size() != model.size()) {
            backend.push_back(backend[0]);
        }

        for (int i = 0; i < model.size(); i++) {
            auto *node = getAiDBNode<AiDBBaseNode>(model_parsing(model[i]));
            if (0 != node->init(model[i], backend[i], zoo)) {
                spdlog::get(AIDB_DEBUG)->error("{} init failed!", model[i]);
            }
            _task_queue.emplace_back(model[i]);
            _task_node.insert(std::pair<std::string, AiDBBaseNode *>(model[i], node));
            _task_backend.insert(std::pair<std::string, std::string>(model[i], backend[i]));
        }

    }

    AiDBFlow::~AiDBFlow() {
        for (auto &it : _task_node) {
            delete it.second;
            it.second = nullptr;
            spdlog::get(AIDB_DEBUG)->debug("AiDBFlow: {} release!", it.first);
        }

    }

    int AiDBFlow::Forward(const unsigned char *frame, int frame_width, int frame_height, json &result) {
        for (const auto &model : _task_queue) {
            auto ret = _task_node[model]->forward(frame, frame_width, frame_height, result);
            if (ret != 0) {
                spdlog::get(AIDB_DEBUG)->error("AiDBFlow Forward  failed, ret: {}", ret);
                return ret;
            }
        }
        return 0;
    }

    std::string AiDBFlow::model_parsing(const std::string &model) {
        if (std::string::npos != model.find("scrfd")) {
            return "AiDBFaceDetectNode";
        } else if (std::string::npos != model.find("pfpld")) {
            return "AiDBFaceLandMarkNode";
        } else if (std::string::npos != model.find("yolov7")) {
            return "AiDBYoloV7Node";
        } else if (std::string::npos != model.find("yolov8")) {
            return "AiDBYoloV8Node";
        } else if (std::string::npos != model.find("yolox")) {
            return "AiDBYoloXNode";
        } else if (std::string::npos != model.find("movenet")) {
            return "AiDBMoveNetNode";
        } else if (std::string::npos != model.find("ppocr_det")) {
            return "AiDBPPDBNetNode";
        } else if (std::string::npos != model.find("ppocr_cls")) {
            return "AiDBPPCLsNode";
        } else if (std::string::npos != model.find("ppocr_ret")) {
            return "AiDBPPCRNNNode";
        } else if (std::string::npos != model.find("mobilevit")) {
            return "AiDBMobileVitNode";
        } else if (std::string::npos != model.find("bisenet")) {
            return "AiDBFaceParsingNode";
        } else if (std::string::npos != model.find("3ddfa")) {
            return "AiDBFace3DDFANode";
        } else if (std::string::npos != model.find("anime")) {
            return "AiDBAnimeGanNode";
        } else {
            std::cout << "Not support\n";
            return "";
        }
    }
}