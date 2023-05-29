// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Performs a clipping operation on all elements of the input node
///
/// All input values that are outside of the <min;max> range are set to 'min' or 'max'
/// depending on which side of the <min;max> range they are. The values that fall into
/// this range remain unchanged.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Clamp : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Clamp", "opset1", UnaryElementwiseArithmetic);

    Clamp() = default;
    /// \brief Constructs a Clamp node.
    ///
    /// \param data - Node producing the input tensor
    /// \param min - the lower bound of the <min;max> range
    /// \param max - the upper bound of the <min;max> range
    Clamp(const Output<Node>& data, const double min, const double max);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    double get_min() const {
        return m_min;
    }
    double get_max() const {
        return m_max;
    }
    void set_min(const double& x) {
        m_min = x;
    }
    void set_max(const double& x) {
        m_max = x;
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool has_evaluate() const override;

private:
    double m_min = 0.0;
    double m_max = 0.0;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
