// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/hswish.hpp"

namespace ngraph {
namespace op {
namespace v4 {
using ov::op::v4::HSwish;
}  // namespace v4
}  // namespace op
}  // namespace ngraph
