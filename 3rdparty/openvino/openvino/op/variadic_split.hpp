// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief VariadicSplit operation splits an input tensor into pieces along some axis.
/// The pieces may have variadic lengths depending on "split_lengths" attribute.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API VariadicSplit : public Op {
public:
    OPENVINO_OP("VariadicSplit", "opset1", op::Op);

    /// \brief Constructs a variadic split operation.
    VariadicSplit() = default;
    /// \brief Constructs a variadic split operation.
    ///
    /// \param data           The tensor to be split.
    /// \param axis           The index of an axis in "data" along which to perform the
    /// split.
    /// \param split_lengths  A list containing the sizes of each output tensor
    /// along the split "axis". Size of "split_lengths" should be equal to the number of
    ///
    /// outputs. The sum of split_lengths must match data.shape[axis]
    VariadicSplit(const Output<Node>& data, const Output<Node>& axis, const Output<Node>& split_lengths);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    size_t get_default_output_index() const override {
        return no_default_index();
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool has_evaluate() const override;
    bool evaluate_label(TensorLabelVector& output_labels) const override;

private:
    bool evaluate_variadic_split(const HostTensorVector& outputs, const HostTensorVector& inputs) const;
    bool has_axis_and_splits_bound_set() const;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
