// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise floor operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Floor : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Floor", "opset1", util::UnaryElementwiseArithmetic);
    /// \brief Constructs a floor operation.
    Floor() = default;
    /// \brief Constructs a floor operation.
    ///
    /// \param arg Node that produces the input tensor.
    Floor(const Output<Node>& arg);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
