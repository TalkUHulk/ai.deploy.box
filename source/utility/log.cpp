//
// Created by TalkUHulk on 2023/3/19.
//

#include "utility/log.h"
#include <string>
namespace AIDB{
    /*
    trace = SPDLOG_LEVEL_TRACE,
    debug = SPDLOG_LEVEL_DEBUG,
    info = SPDLOG_LEVEL_INFO,
    warn = SPDLOG_LEVEL_WARN,
    err = SPDLOG_LEVEL_ERROR,
    critical = SPDLOG_LEVEL_CRITICAL,
    off = SPDLOG_LEVEL_OFF,
    */
    void aidb_log_init(const std::string &logger_name, std::string log_level) {

        if(spdlog::get(logger_name) == nullptr) {
            auto logger = spdlog::stderr_color_mt(logger_name,
                                                  spdlog::color_mode::automatic);
            auto level = spdlog::level::from_str(log_level);
            logger->set_level(spdlog::level::level_enum(level));
            logger->flush_on(spdlog::level::level_enum(level));

        }

    }

    void aidb_log_init(const std::string &logger_name, std::string log_level, std::string file_name,
                       int max_size, int hour, int minute, bool truncate) {

        if(spdlog::get(logger_name) == nullptr) {
            auto logger = spdlog::daily_logger_mt(logger_name, file_name, hour, minute,
                                                  truncate, max_size);
            auto level = spdlog::level::from_str(log_level);
            logger->set_level(spdlog::level::level_enum(level));
            logger->flush_on(spdlog::level::level_enum(level));

        }

    }

}