// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/logical_reduction.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::LogicalReductionKeepDims;
}  // namespace util
}  // namespace op
}  // namespace ngraph
