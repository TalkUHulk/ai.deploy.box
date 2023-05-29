// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset3 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset3_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset3
}  // namespace ngraph
