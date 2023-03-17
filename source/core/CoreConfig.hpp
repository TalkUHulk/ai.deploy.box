//
// Created by TalkUHulk on 2022/11/1.
//

#ifndef AIENGINE_CORE_CONFIG_HPP
#define AIENGINE_CORE_CONFIG_HPP

#ifdef ENABLE_MNN
#include "backend/mnn/MNNParameter.hpp"
#include "backend/mnn/MNNEngine.hpp"
#endif

#ifdef ENABLE_ORT
#include "backend/onnxruntime/ONNXEngine.hpp"
#include "backend/onnxruntime/ONNXParameter.hpp"
#endif

#ifdef ENABLE_NCNN
#include "backend/ncnn/NCNNParameter.hpp"
#include "backend/ncnn/NCNNEngine.hpp"
#endif

#ifdef ENABLE_TNN
#include "backend/tnn/TNNParameter.hpp"
#include "backend/tnn/TNNEngine.hpp"
#endif

#ifdef ENABLE_OPV
#include "backend/openvino/OPVParameter.hpp"
#include "backend/openvino/OPVEngine.hpp"
#endif

#ifdef ENABLE_PPLite
#include "backend/paddle_lite/PPLiteParameter.hpp"
#include "backend/paddle_lite/PPLiteEngine.hpp"
#endif

#endif //AIENGINE_CORE_CONFIG_HPP
