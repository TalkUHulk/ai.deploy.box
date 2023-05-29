// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"

namespace ngraph {
namespace op {
namespace v8 {
using ov::op::v8::AdaptiveAvgPool;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
