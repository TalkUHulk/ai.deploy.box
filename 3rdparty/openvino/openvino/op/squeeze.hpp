// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Squeeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Squeeze : public Op {
public:
    OPENVINO_OP("Squeeze", "opset1");

    Squeeze();
    Squeeze(const Output<Node>& data, const Output<Node>& axes);
    Squeeze(const Output<Node>& data);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_label(TensorLabelVector& output_labels) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool is_dynamic() const override;

private:
    Output<Node> get_default_axes_input() const;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
