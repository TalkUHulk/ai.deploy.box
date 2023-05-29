// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/round.hpp"

namespace ngraph {
namespace op {
namespace v5 {
using ov::op::v5::Round;
}  // namespace v5
}  // namespace op
}  // namespace ngraph
