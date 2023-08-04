//
// Created by TalkUHulk on 2023/6/29.
//

#include "aidb_server.hpp"
#include <opencv2/opencv.hpp>
#include "utility/log.h"
#include "cpp-base64/base64.h"

namespace AiDBServer {
    AiDBServer::~AiDBServer() {
        for (auto &it : _task_flow) {
            spdlog::get(AIDB_DEBUG)->debug("AiDBServer task flow[{}] release", it.first);
            delete it.second;
            it.second = nullptr;
        }
    }

    int AiDBServer::Register(const char *parameter) {
        auto input = json::parse(parameter);
        auto model = input["model"];
        auto backend = input["backend"];
        auto flow_uuid = input["flow_uuid"];
        if (backend.size() != model.size() && backend.size() != 1) {
            spdlog::get(AIDB_DEBUG)->error("Input Parameter Invalid, backend size != model size");
            return -1;
        }

        if(_task_flow.find(flow_uuid) != _task_flow.end()){
            spdlog::get(AIDB_DEBUG)->debug("task flow: [{}] has existed.", flow_uuid);
            return -2;
        }
        while (backend.size() != model.size()) {
            backend.push_back(backend[0]);
        }

        auto *task_flow = new AiDBFlow(input);
        for (int i = 0; i < model.size(); i++) {
            _task_flow.insert(std::pair<std::string, AiDBFlow *>(flow_uuid, task_flow));
        }
        spdlog::get(AIDB_DEBUG)->debug("task flow: [{}] register.", flow_uuid);
        return 0;
    }

    int AiDBServer::unRegister(const char *flow_uuid) {
        auto str_flow_uuid = std::string(flow_uuid);
        auto iter = _task_flow.find(str_flow_uuid);
        if (iter != _task_flow.end()) {
            delete iter->second;
            iter->second = nullptr;
            _task_flow.erase(str_flow_uuid);
            spdlog::get(AIDB_DEBUG)->debug("task flow: [{}] register succeed.", str_flow_uuid);
            return 0;
        } else {
            spdlog::get(AIDB_DEBUG)->error("task flow: [{}] unregister failed, task flow not exists", str_flow_uuid);
            return -1;
        }

    }


    int AiDBServer::Forward(const char *binary_image, const char *flow_uuid, char *binary_result, int size_in,
                            int *size_out) {

        auto str_flow_uuid = std::string(flow_uuid);
        auto iter = _task_flow.find(str_flow_uuid);
        if (iter != _task_flow.end()) {
            std::string decoded = base64_decode(binary_image);
            std::vector<char> im_vec(decoded.begin(),decoded.end());
            auto image = cv::imdecode(im_vec, 1);
//            cv::imwrite("receive.jpg", image);
//            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                spdlog::get(AIDB_DEBUG)->error("AiDBServer read image failed");
                return -1;
            }

            json aidb_result;
            aidb_result["error_code"] = 0;
            json face_metas_output = json::array();

            auto ret = iter->second->Forward(image.data, image.cols, image.rows, aidb_result);
            if (0 != ret) {
                spdlog::get(AIDB_DEBUG)->error("AiDBServer forward failed, ret:{}", ret);
                return ret;
            }
            auto str_aidb_result = aidb_result.dump();

            *size_out = str_aidb_result.size() * sizeof(char) + 1;
            if (*size_out > size_in) {
                spdlog::get(AIDB_DEBUG)->error("size_in < {}", *size_out);
                return -2;
            }

            strncpy(binary_result, str_aidb_result.data(), *size_out);
            binary_result[*size_out] = '\0';

            return ret;
        } else {
            spdlog::get(AIDB_DEBUG)->error("AiDBServer Forward: task flow:[{}] not exists.", str_flow_uuid);
            return -1;
        }
    }

    AiDBServer::AiDBServer() {
        AIDB::aidb_log_init(AIDB_DEBUG, "debug");
    }
}

