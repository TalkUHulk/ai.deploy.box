// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/atanh.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::Atanh;
}  // namespace v3
using v3::Atanh;
}  // namespace op
}  // namespace ngraph
