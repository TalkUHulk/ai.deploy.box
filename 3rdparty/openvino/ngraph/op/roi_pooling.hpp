// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/roi_pooling.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::ROIPooling;
}  // namespace v0
using v0::ROIPooling;
}  // namespace op
}  // namespace ngraph
