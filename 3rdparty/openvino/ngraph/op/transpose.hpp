// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/transpose.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Transpose;
}  // namespace v1
using v1::Transpose;
}  // namespace op
}  // namespace ngraph
