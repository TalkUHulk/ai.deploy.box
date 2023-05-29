// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/matmul.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::MatMul;
}  // namespace v0
using v0::MatMul;
}  // namespace op
}  // namespace ngraph
