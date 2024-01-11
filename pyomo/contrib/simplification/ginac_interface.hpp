#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <ginac/ginac.h>
#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace py = pybind11;
using namespace pybind11::literals;
using namespace GiNaC;

enum ExprType {
  py_float = 0,
  var = 1,
  param = 2,
  product = 3,
  sum = 4,
  negation = 5,
  external_func = 6,
  power = 7,
  division = 8,
  unary_func = 9,
  linear = 10,
  named_expr = 11,
  numeric_constant = 12,
  pyomo_unit = 13,
  unary_abs = 14
};

class PyomoExprTypes {
public:
  PyomoExprTypes() {
    expr_type_map[int_] = py_float;
    expr_type_map[float_] = py_float;
    expr_type_map[np_int16] = py_float;
    expr_type_map[np_int32] = py_float;
    expr_type_map[np_int64] = py_float;
    expr_type_map[np_longlong] = py_float;
    expr_type_map[np_uint16] = py_float;
    expr_type_map[np_uint32] = py_float;
    expr_type_map[np_uint64] = py_float;
    expr_type_map[np_ulonglong] = py_float;
    expr_type_map[np_float16] = py_float;
    expr_type_map[np_float32] = py_float;
    expr_type_map[np_float64] = py_float;
    expr_type_map[ScalarVar] = var;
    expr_type_map[_GeneralVarData] = var;
    expr_type_map[AutoLinkedBinaryVar] = var;
    expr_type_map[ScalarParam] = param;
    expr_type_map[_ParamData] = param;
    expr_type_map[MonomialTermExpression] = product;
    expr_type_map[ProductExpression] = product;
    expr_type_map[NPV_ProductExpression] = product;
    expr_type_map[SumExpression] = sum;
    expr_type_map[NPV_SumExpression] = sum;
    expr_type_map[NegationExpression] = negation;
    expr_type_map[NPV_NegationExpression] = negation;
    expr_type_map[ExternalFunctionExpression] = external_func;
    expr_type_map[NPV_ExternalFunctionExpression] = external_func;
    expr_type_map[PowExpression] = ExprType::power;
    expr_type_map[NPV_PowExpression] = ExprType::power;
    expr_type_map[DivisionExpression] = division;
    expr_type_map[NPV_DivisionExpression] = division;
    expr_type_map[UnaryFunctionExpression] = unary_func;
    expr_type_map[NPV_UnaryFunctionExpression] = unary_func;
    expr_type_map[LinearExpression] = linear;
    expr_type_map[_GeneralExpressionData] = named_expr;
    expr_type_map[ScalarExpression] = named_expr;
    expr_type_map[Integral] = named_expr;
    expr_type_map[ScalarIntegral] = named_expr;
    expr_type_map[NumericConstant] = numeric_constant;
    expr_type_map[_PyomoUnit] = pyomo_unit;
    expr_type_map[AbsExpression] = unary_abs;
    expr_type_map[NPV_AbsExpression] = unary_abs;
  }
  ~PyomoExprTypes() = default;
  py::int_ ione = 1;
  py::float_ fone = 1.0;
  py::type int_ = py::type::of(ione);
  py::type float_ = py::type::of(fone);
  py::object np = py::module_::import("numpy");
  py::type np_int16 = np.attr("int16");
  py::type np_int32 = np.attr("int32");
  py::type np_int64 = np.attr("int64");
  py::type np_longlong = np.attr("longlong");
  py::type np_uint16 = np.attr("uint16");
  py::type np_uint32 = np.attr("uint32");
  py::type np_uint64 = np.attr("uint64");
  py::type np_ulonglong = np.attr("ulonglong");
  py::type np_float16 = np.attr("float16");
  py::type np_float32 = np.attr("float32");
  py::type np_float64 = np.attr("float64");
  py::object ScalarParam =
      py::module_::import("pyomo.core.base.param").attr("ScalarParam");
  py::object _ParamData =
      py::module_::import("pyomo.core.base.param").attr("_ParamData");
  py::object ScalarVar =
      py::module_::import("pyomo.core.base.var").attr("ScalarVar");
  py::object _GeneralVarData =
      py::module_::import("pyomo.core.base.var").attr("_GeneralVarData");
  py::object AutoLinkedBinaryVar =
      py::module_::import("pyomo.gdp.disjunct").attr("AutoLinkedBinaryVar");
  py::object numeric_expr = py::module_::import("pyomo.core.expr.numeric_expr");
  py::object NegationExpression = numeric_expr.attr("NegationExpression");
  py::object NPV_NegationExpression =
      numeric_expr.attr("NPV_NegationExpression");
  py::object ExternalFunctionExpression =
      numeric_expr.attr("ExternalFunctionExpression");
  py::object NPV_ExternalFunctionExpression =
      numeric_expr.attr("NPV_ExternalFunctionExpression");
  py::object PowExpression = numeric_expr.attr("PowExpression");
  py::object NPV_PowExpression = numeric_expr.attr("NPV_PowExpression");
  py::object ProductExpression = numeric_expr.attr("ProductExpression");
  py::object NPV_ProductExpression = numeric_expr.attr("NPV_ProductExpression");
  py::object MonomialTermExpression =
      numeric_expr.attr("MonomialTermExpression");
  py::object DivisionExpression = numeric_expr.attr("DivisionExpression");
  py::object NPV_DivisionExpression =
      numeric_expr.attr("NPV_DivisionExpression");
  py::object SumExpression = numeric_expr.attr("SumExpression");
  py::object NPV_SumExpression = numeric_expr.attr("NPV_SumExpression");
  py::object UnaryFunctionExpression =
      numeric_expr.attr("UnaryFunctionExpression");
  py::object AbsExpression = numeric_expr.attr("AbsExpression");
  py::object NPV_AbsExpression = numeric_expr.attr("NPV_AbsExpression");
  py::object NPV_UnaryFunctionExpression =
      numeric_expr.attr("NPV_UnaryFunctionExpression");
  py::object LinearExpression = numeric_expr.attr("LinearExpression");
  py::object NumericConstant =
      py::module_::import("pyomo.core.expr.numvalue").attr("NumericConstant");
  py::object expr_module = py::module_::import("pyomo.core.base.expression");
  py::object _GeneralExpressionData =
      expr_module.attr("_GeneralExpressionData");
  py::object ScalarExpression = expr_module.attr("ScalarExpression");
  py::object ScalarIntegral =
      py::module_::import("pyomo.dae.integral").attr("ScalarIntegral");
  py::object Integral =
      py::module_::import("pyomo.dae.integral").attr("Integral");
  py::object _PyomoUnit =
      py::module_::import("pyomo.core.base.units_container").attr("_PyomoUnit");
  py::object exp = numeric_expr.attr("exp");
  py::object log = numeric_expr.attr("log");
  py::object sin = numeric_expr.attr("sin");
  py::object cos = numeric_expr.attr("cos");
  py::object tan = numeric_expr.attr("tan");
  py::object asin = numeric_expr.attr("asin");
  py::object acos = numeric_expr.attr("acos");
  py::object atan = numeric_expr.attr("atan");
  py::object sqrt = numeric_expr.attr("sqrt");
  py::object builtins = py::module_::import("builtins");
  py::object id = builtins.attr("id");
  py::object len = builtins.attr("len");
  py::dict expr_type_map;
};

ex pyomo_to_ginac(py::handle expr, PyomoExprTypes &expr_types);


class GinacInterface {
    public:
    std::unordered_map<long, ex> leaf_map;
    std::unordered_map<ex, py::object> ginac_pyomo_map;
    PyomoExprTypes expr_types;
    bool symbolic_solver_labels = false;

    GinacInterface() = default;
    GinacInterface(bool _symbolic_solver_labels) : symbolic_solver_labels(_symbolic_solver_labels) {}
    ~GinacInterface() = default;

    ex to_ginac(py::handle expr);
    py::object from_ginac(ex &ginac_expr);
};
