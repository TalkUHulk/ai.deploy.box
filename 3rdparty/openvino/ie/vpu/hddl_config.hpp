// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for HDDL plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file hddl_config.hpp
 */

#pragma once

#include "vpu_config.hpp"

namespace InferenceEngine {

namespace Metrics {

/**
 * @brief Metric to get a int of the device number, String value is METRIC_HDDL_DEVICE_NUM
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_NUM, int);

/**
 * @brief Metric to get a std::vector<std::string> of device names, String value is METRIC_HDDL_DEVICE_NAME
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_NAME, std::vector<std::string>);

/**
 * @brief  Metric to get a std::vector<float> of device thermal, String value is METRIC_HDDL_DEVICE_THERMAL
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_THERMAL, std::vector<float>);

/**
 * @brief  Metric to get a std::vector<uint32> of device ids, String value is METRIC_HDDL_DEVICE_ID
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_ID, std::vector<unsigned int>);

/**
 * @brief  Metric to get a std::vector<int> of device subclasses, String value is METRIC_HDDL_DEVICE_SUBCLASS
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_SUBCLASS, std::vector<int>);

/**
 * @brief  Metric to get a std::vector<uint32> of device total memory, String value is METRIC_HDDL_MEMORY_TOTAL
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_MEMORY_TOTAL, std::vector<unsigned int>);

/**
 * @brief  Metric to get a std::vector<uint32> of device used memory, String value is METRIC_HDDL_DEVICE_MEMORY_USED
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_MEMORY_USED, std::vector<unsigned int>);

/**
 * @brief  Metric to get a std::vector<float> of device utilization, String value is METRIC_HDDL_DEVICE_UTILIZATION
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_UTILIZATION, std::vector<float>);

/**
 * @brief  Metric to get a std::vector<std::string> of stream ids, String value is METRIC_HDDL_DEVICE_STREAM_ID
 */
DECLARE_METRIC_KEY(HDDL_STREAM_ID, std::vector<std::string>);

/**
 * @brief  Metric to get a std::vector<std::string> of device tags, String value is METRIC_HDDL_DEVICE_TAG
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_TAG, std::vector<std::string>);

/**
 * @brief  Metric to get a std::vector<int> of group ids, String value is METRIC_HDDL_GROUP_ID
 */
DECLARE_METRIC_KEY(HDDL_GROUP_ID, std::vector<int>);

/**
 * @brief  Metric to get a int number of device be using for group, String value is METRIC_HDDL_DEVICE_GROUP_USING_NUM
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_GROUP_USING_NUM, int);

/**
 * @brief  Metric to get a int number of total device, String value is METRIC_HDDL_DEVICE_TOTAL_NUM
 */
DECLARE_METRIC_KEY(HDDL_DEVICE_TOTAL_NUM, int);

}  // namespace Metrics

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This option allows to specify the number of MYX devices used for inference a specific Executable network.
 * Note: Only one network would be allocated to one device.
 * The number of devices for the tag is specified in the hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "graph_tag_map":
 *     {
 *         "tagA":3
 *     }
 * }
 * It means that an executable network marked with tagA will be executed on 3 devices
 */
DECLARE_VPU_CONFIG(HDDL_GRAPH_TAG);

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This config makes the executable networks to be allocated on one certain device (instead of multiple devices).
 * And all inference through this executable network, will be done on this device.
 * Note: Only one network would be allocated to one device.
 * The number of devices which will be used for stream-affinity must be specified in hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "stream_device_number":5
 * }
 * It means that 5 device will be used for stream-affinity
 */
DECLARE_VPU_CONFIG(HDDL_STREAM_ID);

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This config allows user to control device flexibly. This config gives a "tag" for a certain device while
 * allocating a network to it. Afterward, user can allocating/deallocating networks to this device with this "tag".
 * Devices used for such use case is controlled by a so-called "Bypass Scheduler" in HDDL backend, and the number
 * of such device need to be specified in hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "bypass_device_number": 5
 * }
 * It means that 5 device will be used for Bypass scheduler.
 */
DECLARE_VPU_CONFIG(HDDL_DEVICE_TAG);

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: "YES/NO", default is "NO".
 * This config is a sub-config of DEVICE_TAG, and only available when "DEVICE_TAG" is set. After a user load a
 * network, the user got a handle for the network.
 * If "YES", the network allocated is bind to the device (with the specified "DEVICE_TAG"), which means all afterwards
 * inference through this network handle will be executed on this device only.
 * If "NO", the network allocated is not bind to the device (with the specified "DEVICE_TAG"). If the same network
 * is allocated on multiple other devices (also set BIND_DEVICE to "False"), then inference through any handle of these
 * networks may be executed on any of these devices those have the network loaded.
 */
DECLARE_VPU_CONFIG(HDDL_BIND_DEVICE);

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: A signed int wrapped in a string, default is "0".
 * This config is a sub-config of DEVICE_TAG, and only available when "DEVICE_TAG" is set and "BIND_DEVICE" is "False".
 * When there are multiple devices running a certain network (a same network running on multiple devices in Bypass
 * Scheduler), the device with a larger number has a higher priority, and more inference tasks will be fed to it with
 * priority.
 */
DECLARE_VPU_CONFIG(HDDL_RUNTIME_PRIORITY);

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: "YES/NO", default is "NO".
 * SGAD is short for "Single Graph All Device". With this scheduler, once application allocates 1 network, all devices
 * (managed by SGAD scheduler) will be loaded with this graph. The number of network that can be loaded to one device
 * can exceed one. Once application deallocates 1 network from device, all devices will unload the network from them.
 */
DECLARE_VPU_CONFIG(HDDL_USE_SGAD);

/**
 * @brief [Only for OpenVINO Intel HDDL device]
 * Type: A signed int wrapped in a string, default is "0".
 * This config gives a "group id" for a certain device when this device has been reserved for certain client, client
 * can use this device grouped by calling this group id while other client can't use this device
 * Each device has their own group id. Device in one group shares same group id.
 */
DECLARE_VPU_CONFIG(HDDL_GROUP_DEVICE);

}  // namespace InferenceEngine
