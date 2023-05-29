// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace op {
namespace util {
struct VariableInfo {
    PartialShape data_shape;
    element::Type data_type;
    std::string variable_id;

    inline bool operator==(const VariableInfo& other) const {
        return data_shape == other.data_shape && data_type == other.data_type && variable_id == other.variable_id;
    }
};

class OPENVINO_API Variable {
public:
    using Ptr = std::shared_ptr<Variable>;
    Variable() = default;

    explicit Variable(VariableInfo variable_info) : m_info(std::move(variable_info)) {}

    VariableInfo get_info() const {
        return m_info;
    }
    void update(const VariableInfo& variable_info) {
        m_info = variable_info;
    }

private:
    VariableInfo m_info;
};
using VariableVector = std::vector<Variable::Ptr>;

}  // namespace util
}  // namespace op
template <>
class OPENVINO_API AttributeAdapter<std::shared_ptr<op::util::Variable>>
    : public DirectValueAccessor<std::shared_ptr<op::util::Variable>> {
public:
    explicit AttributeAdapter(std::shared_ptr<op::util::Variable>& value)
        : DirectValueAccessor<std::shared_ptr<op::util::Variable>>(value) {}

    OPENVINO_RTTI("AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>");
};
}  // namespace ov
