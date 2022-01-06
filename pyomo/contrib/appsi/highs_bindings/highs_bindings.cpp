#include "Highs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(appsi_highs, m)
{
  py::enum_<HighsStatus>(m, "HighsStatus")
    .value("kError", HighsStatus::kError)
    .value("kOk", HighsStatus::kOk)
    .value("kWarning", HighsStatus::kWarning);
  py::class_<HighsSolution>(m, "HighsSolution")
    .def_readwrite("value_valid", &HighsSolution::value_valid)
    .def_readwrite("dual_valid", &HighsSolution::dual_valid)
    .def_readwrite("col_value", &HighsSolution::col_value)
    .def_readwrite("col_dual", &HighsSolution::col_dual)
    .def_readwrite("row_value", &HighsSolution::row_value)
    .def_readwrite("row_dual", &HighsSolution::row_dual);
  py::class_<Highs>(m, "Highs")
    .def(py::init<>())
    .def("run", &Highs::run)
    .def("getSolution", &Highs::getSolution)
    .def("changeObjectiveSense", &Highs::changeObjectiveSense)
    .def("changeObjectiveOffset", &Highs::changeObjectiveOffset)
    .def("changeColIntegrality", &Highs::changeColIntegrality)
    .def("changeColCost", &Highs::changeColCost)
    .def("changeColBounds", &Highs::changeColBounds)
    .def("changeRowBounds", &Highs::changeRowBounds)
    .def("changeCoeff", &Highs::changeCoeff)
    .def("addRows", &Highs::addRows)
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const bool)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const int)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const double)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const std::string)>(&Highs::setOptionValue));
}
