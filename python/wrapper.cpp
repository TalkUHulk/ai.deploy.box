//
// Created by TalkUHulk on 2023/6/12.
//

#include <pybind11/pybind11.h>
#include "wrapper/AiDBWrapper.hpp"

namespace py = pybind11;


PYBIND11_MODULE(pyAiDB, m)
{
    py::class_<AiDBWrapper>(m, "AiDB", py::dynamic_attr())
            .def(py::init<>())
            .def("init", &AiDBWrapper::init)
            .def("forward", py::overload_cast<py::array_t<uint8_t>&, int, int>(&AiDBWrapper::forward), py::return_value_policy::move)
            .def("forward", py::overload_cast<py::array_t<float>&>(&AiDBWrapper::forward), py::return_value_policy::move);


    py::enum_<AIDB_ModelID>(m, "AIDB_ModelID")
            .value("EMPTY", EMPTY)
            .value("SCRFD", SCRFD)
            .value("PFPLD", PFPLD)
            .value("TDDDFA", TDDDFA)
            .value("BISENET", BISENET)
            .value("MOVENET", MOVENET)
            .value("YOLOX", YOLOX)
            .value("YOLOV7", YOLOV7)
            .value("YOLOV8", YOLOV8)
            .value("PPOCR", PPOCR)
            .value("MOBILE_VIT", MOBILE_VIT)
            .value("MOBILE_STYLE_GAN", MOBILE_STYLE_GAN)
            .value("ANIME_GAN", ANIME_GAN)
            .export_values();

}