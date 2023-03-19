//
// Created by TalkUHulk on 2023/3/19.
//

#ifndef AIDEPLOYBOX_LOG_H
#define AIDEPLOYBOX_LOG_H

#include "spdlog/logger.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace AIDB{
    void aidb_log_init(const std::string &logger_name, std::string log_level, std::string file_name,
                       int max_size=104857600, int hour=0, int minute=0, bool truncate=true);

    void aidb_log_init(const std::string &logger_name, std::string log_level);
}
#endif //AIDEPLOYBOX_LOG_H
