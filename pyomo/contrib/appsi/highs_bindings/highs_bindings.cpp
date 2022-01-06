#include "Highs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace pybind11::literals;


void highs_addRow(Highs* h, double lower, double upper, int num_new_nz, py::array_t<int> indices, py::array_t<double> values)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info values_info = values.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->addRow(lower, upper, num_new_nz, indices_ptr, values_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding row");
}


void highs_addRows(Highs* h, int num_cons, py::array_t<double> lower, py::array_t<double> upper, int num_new_nz,
		   py::array_t<int> starts, py::array_t<int> indices, py::array_t<double> values)
{
  py::buffer_info lower_info = lower.request();
  py::buffer_info upper_info = upper.request();
  py::buffer_info starts_info = starts.request();
  py::buffer_info indices_info = indices.request();
  py::buffer_info values_info = values.request();

  double* lower_ptr = static_cast<double*>(lower_info.ptr);
  double* upper_ptr = static_cast<double*>(upper_info.ptr);
  int* starts_ptr = static_cast<int*>(starts_info.ptr);
  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->addRows(num_cons, lower_ptr, upper_ptr, num_new_nz, starts_ptr, indices_ptr, values_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding rows");
}


void highs_addCol(Highs* h, double cost, double lower, double upper, int num_new_nz, py::array_t<int> indices, py::array_t<double> values)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info values_info = values.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->addCol(cost, lower, upper, num_new_nz, indices_ptr, values_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding col");
}


void highs_addVar(Highs* h, double lower, double upper)
{
  int* indices;
  double* values;

  HighsStatus status = h->addCol(0, lower, upper, 0, indices, values);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding var");  
}


void highs_addVars(Highs* h, int num_vars, py::array_t<double> lower, py::array_t<double> upper)
{
  int* starts;
  int* indices;
  double* values;
  double* costs = new double[num_vars]();

  py::buffer_info lower_info = lower.request();
  py::buffer_info upper_info = upper.request();

  double* lower_ptr = static_cast<double*>(lower_info.ptr);
  double* upper_ptr = static_cast<double*>(upper_info.ptr);

  HighsStatus status = h->addCols(num_vars, costs, lower_ptr, upper_ptr, 0, starts, indices, values);

  delete[] costs;

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding vars");  
}


void highs_changeColsCost(Highs* h, int num_set_entries, py::array_t<int> indices, py::array_t<double> cost)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info cost_info = cost.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* cost_ptr = static_cast<double*>(cost_info.ptr);

  HighsStatus status = h->changeColsCost(num_set_entries, indices_ptr, cost_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when changing objective coefficients");  
}


void highs_changeColsBounds(Highs* h, int num_set_entries, py::array_t<int> indices, py::array_t<double> lower, py::array_t<double> upper)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info lower_info = lower.request();
  py::buffer_info upper_info = upper.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* lower_ptr = static_cast<double*>(lower_info.ptr);
  double* upper_ptr = static_cast<double*>(upper_info.ptr);

  HighsStatus status = h->changeColsBounds(num_set_entries, indices_ptr, lower_ptr, upper_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when changing variable bounds");  
}


void highs_changeColsIntegrality(Highs* h, int num_set_entries, py::array_t<int> indices, py::array_t<HighsVarType> integrality)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info integrality_info = integrality.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  HighsVarType* integrality_ptr = static_cast<HighsVarType*>(integrality_info.ptr);

  HighsStatus status = h->changeColsIntegrality(num_set_entries, indices_ptr, integrality_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when changing variable integrality");  
}


void highs_deleteVars(Highs* h, int num_set_entries, py::array_t<int> indices)
{
  py::buffer_info indices_info = indices.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);

  HighsStatus status = h->deleteCols(num_set_entries, indices_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when deleting columns");  
}


void highs_deleteRows(Highs* h, int num_set_entries, py::array_t<int> indices)
{
  py::buffer_info indices_info = indices.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);

  HighsStatus status = h->deleteRows(num_set_entries, indices_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when deleting rows");  
}


PYBIND11_MODULE(appsi_highs, m)
{
  py::enum_<ObjSense>(m, "ObjSense")
    .value("kMinimize", ObjSense::kMinimize)
    .value("kMaximize", ObjSense::kMaximize);
  py::enum_<HighsModelStatus>(m, "HighsModelStatus")
    .value("kNotset", HighsModelStatus::kNotset)
    .value("kLoadError", HighsModelStatus::kLoadError)
    .value("kModelError", HighsModelStatus::kModelError)
    .value("kPresolveError", HighsModelStatus::kPresolveError)
    .value("kSolveError", HighsModelStatus::kSolveError)
    .value("kPostsolveError", HighsModelStatus::kPostsolveError)
    .value("kModelEmpty", HighsModelStatus::kModelEmpty)
    .value("kOptimal", HighsModelStatus::kOptimal)
    .value("kInfeasible", HighsModelStatus::kInfeasible)
    .value("kUnboundedOrInfeasible", HighsModelStatus::kUnboundedOrInfeasible)
    .value("kUnbounded", HighsModelStatus::kUnbounded)
    .value("kObjectiveBound", HighsModelStatus::kObjectiveBound)
    .value("kObjectiveTarget", HighsModelStatus::kObjectiveTarget)
    .value("kTimeLimit", HighsModelStatus::kTimeLimit)
    .value("kIterationLimit", HighsModelStatus::kIterationLimit)
    .value("kUnknown", HighsModelStatus::kUnknown);
  py::enum_<HighsVarType>(m, "HighsVarType")
    .value("kContinuous", HighsVarType::kContinuous)
    .value("kInteger", HighsVarType::kInteger);
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
    .def("getObjectiveValue", &Highs::getObjectiveValue)
    .def("getRunTime", &Highs::getRunTime)
    .def("getModelStatus", &Highs::getModelStatus)
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const bool)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const int)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const double)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const std::string)>(&Highs::setOptionValue));
  m.def("highs_addRow", &highs_addRow);
  m.def("highs_addRows", &highs_addRows);
  m.def("highs_addCol", &highs_addCol);
  m.def("highs_addVar", &highs_addVar);
  m.def("highs_addVars", &highs_addVars);
  m.def("highs_changeColsCost", &highs_changeColsCost);
  m.def("highs_changeColsBounds", &highs_changeColsBounds);
  m.def("highs_changeColsIntegrality", &highs_changeColsIntegrality);
  m.def("highs_deleteVars", &highs_deleteVars);
  m.def("highs_deleteRows", &highs_deleteRows);
  m.attr("kHighsInf") = kHighsInf;
  m.attr("HIGHS_VERSION_MAJOR") = HIGHS_VERSION_MAJOR;
  m.attr("HIGHS_VERSION_MINOR") = HIGHS_VERSION_MINOR;
  m.attr("HIGHS_VERSION_PATCH") = HIGHS_VERSION_PATCH;
}
