// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a
///        bounding box, optionally with stride.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API StridedSlice : public Op {
public:
    OPENVINO_OP("StridedSlice", "opset1", op::Op);

    StridedSlice() = default;

    /// \brief Constructs a dynamic tensor strided slice operation.
    ///
    /// \param data             The tensor to be sliced.
    /// \param begin            1D tensor with begin indexes for input blob slicing.
    /// \param end              1D tensor with end indexes for input blob slicing.
    /// \param strides          The slicing strides; for example, strides of `{n,m}`
    ///                         means to take every nth row and every mth column
    ///                         of the input matrix.
    /// \param begin_mask       When begin_mask[i] equal to 1 means that the
    ///                         corresponding dimension of the begin input is ignored.
    /// \param end_mask         When end_mask[i] is 1, the corresponding dimension of
    ///                         the end input is ignored.
    /// \param new_axis_mask    If new_axis_mask[i] is 1, a length 1 dimension
    ///                         is inserted on the i-th position.
    /// \param shrink_axis_mask If shrink_axis_mask[i] is 1, the dimension
    ///                         on the i-th position is deleted.
    /// \param ellipsis_mask    It inserts missing dimensions
    ///                         on a position of a non-zero bit.
    StridedSlice(const Output<Node>& data,
                 const Output<Node>& begin,
                 const Output<Node>& end,
                 const Output<Node>& strides,
                 const std::vector<int64_t>& begin_mask,
                 const std::vector<int64_t>& end_mask,
                 const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                 const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                 const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

    /// \brief Constructs a dynamic tensor strided slice operation.
    ///
    /// \param data             The tensor to be sliced.
    /// \param begin            1D tensor with begin indexes for input blob slicing.
    /// \param end              1D tensor with end indexes for input blob slicing.
    /// \param begin_mask       When begin_mask[i] equal to 1 means that the
    ///                         corresponding dimension of the begin input is ignored.
    /// \param end_mask         When end_mask[i] is 1, the corresponding dimension of
    ///                         the end input is ignored.
    /// \param new_axis_mask    If new_axis_mask[i] is 1, a length 1 dimension
    ///                         is inserted on the i-th position.
    /// \param shrink_axis_mask If shrink_axis_mask[i] is 1, the dimension
    ///                         on the i-th position is deleted.
    /// \param ellipsis_mask    It inserts missing dimensions
    ///                         on a position of a non-zero bit.
    StridedSlice(const Output<Node>& data,
                 const Output<Node>& begin,
                 const Output<Node>& end,
                 const std::vector<int64_t>& begin_mask,
                 const std::vector<int64_t>& end_mask,
                 const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                 const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                 const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

    bool visit_attributes(AttributeVisitor& visitor) override;
    const std::vector<int64_t>& get_begin_mask() const {
        return m_begin_mask;
    }
    void set_begin_mask(const std::vector<int64_t>& vec) {
        m_begin_mask = vec;
    }
    const std::vector<int64_t>& get_end_mask() const {
        return m_end_mask;
    }
    void set_end_mask(const std::vector<int64_t>& vec) {
        m_end_mask = vec;
    }
    const std::vector<int64_t>& get_new_axis_mask() const {
        return m_new_axis_mask;
    }
    void set_new_axis_mask(const std::vector<int64_t>& vec) {
        m_new_axis_mask = vec;
    }
    const std::vector<int64_t>& get_shrink_axis_mask() const {
        return m_shrink_axis_mask;
    }
    void set_shrink_axis_mask(const std::vector<int64_t>& vec) {
        m_shrink_axis_mask = vec;
    }
    const std::vector<int64_t>& get_ellipsis_mask() const {
        return m_ellipsis_mask;
    }
    void set_ellipsis_mask_mask(const std::vector<int64_t>& vec) {
        m_ellipsis_mask = vec;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_label(TensorLabelVector& output_labels) const override;

private:
    AxisSet convert_mask_to_axis_set(const std::vector<int64_t>& mask) const;

    std::vector<int64_t> m_begin_mask;
    std::vector<int64_t> m_end_mask;
    std::vector<int64_t> m_new_axis_mask;
    std::vector<int64_t> m_shrink_axis_mask;
    std::vector<int64_t> m_ellipsis_mask;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
