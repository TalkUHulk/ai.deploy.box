// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"

namespace ngraph {
namespace op {
namespace v6 {
using ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage;
}  // namespace v6
}  // namespace op
}  // namespace ngraph
