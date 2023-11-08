#include "ginac_interface.hpp"

ex ginac_expr_from_pyomo_node(py::handle expr, std::unordered_map<long, ex> &leaf_map, PyomoExprTypes &expr_types) {
  ex res;
  ExprType tmp_type =
      expr_types.expr_type_map[py::type::of(expr)].cast<ExprType>();

  switch (tmp_type) {
  case py_float: {
    res = numeric(expr.cast<double>());
    break;
  }
  case var: {
    long expr_id = expr_types.id(expr).cast<long>();
    if (leaf_map.count(expr_id) == 0) {
      leaf_map[expr_id] = symbol("x" + std::to_string(expr_id));
    }
    res = leaf_map[expr_id];
    break;
  }
  case param: {
    long expr_id = expr_types.id(expr).cast<long>();
    if (leaf_map.count(expr_id) == 0) {
      leaf_map[expr_id] = symbol("p" + std::to_string(expr_id));
    }
    res = leaf_map[expr_id];
    break;
  }
  case product: {
    py::list pyomo_args = expr.attr("args");
    res = ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types) * ginac_expr_from_pyomo_node(pyomo_args[1], leaf_map, expr_types);
    break;
  }
  case sum: {
    py::list pyomo_args = expr.attr("args");
    for (py::handle arg : pyomo_args) {
      res += ginac_expr_from_pyomo_node(arg, leaf_map, expr_types);
    }
    break;
  }
  case negation: {
    py::list pyomo_args = expr.attr("args");
    res = - ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types);
    break;
  }
  case external_func: {
    long expr_id = expr_types.id(expr).cast<long>();
    if (leaf_map.count(expr_id) == 0) {
      leaf_map[expr_id] = symbol("f" + std::to_string(expr_id));
    }
    res = leaf_map[expr_id];
    break;
  }
  case ExprType::power: {
    py::list pyomo_args = expr.attr("args");
    res = pow(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types), ginac_expr_from_pyomo_node(pyomo_args[1], leaf_map, expr_types));
    break;
  }
  case division: {
    py::list pyomo_args = expr.attr("args");
    res = ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types) / ginac_expr_from_pyomo_node(pyomo_args[1], leaf_map, expr_types);
    break;
  }
  case unary_func: {
    std::string function_name = expr.attr("getname")().cast<std::string>();
    py::list pyomo_args = expr.attr("args");
    if (function_name == "exp")
      res = exp(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "log")
      res = log(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "sin")
      res = sin(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "cos")
      res = cos(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "tan")
      res = tan(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "asin")
      res = asin(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "acos")
      res = acos(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "atan")
      res = atan(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else if (function_name == "sqrt")
      res = sqrt(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    else
      throw py::value_error("Unrecognized expression type: " + function_name);
    break;
  }
  case linear: {
    py::list pyomo_args = expr.attr("args");
    for (py::handle arg : pyomo_args) {
      res += ginac_expr_from_pyomo_node(arg, leaf_map, expr_types);
    }
    break;
  }
  case named_expr: {
    res = ginac_expr_from_pyomo_node(expr.attr("expr"), leaf_map, expr_types);
    break;
  }
  case numeric_constant: {
    res = numeric(expr.attr("value").cast<double>());
    break;
  }
  case pyomo_unit: {
    res = numeric(1.0);
    break;
  }
  case unary_abs: {
    py::list pyomo_args = expr.attr("args");
    res = abs(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, expr_types));
    break;
  }
  default: {
    throw py::value_error("Unrecognized expression type: " +
                          expr_types.builtins.attr("str")(py::type::of(expr))
                              .cast<std::string>());
    break;
  }
  }
  return res;
}

ex ginac_expr_from_pyomo_expr(py::handle expr, PyomoExprTypes &expr_types) {
  std::unordered_map<long, ex> leaf_map;
  ex res = ginac_expr_from_pyomo_node(expr, leaf_map, expr_types);
  return res;
}


PYBIND11_MODULE(ginac_interface, m) {
  m.def("ginac_expr_from_pyomo_expr", &ginac_expr_from_pyomo_expr);
  py::class_<PyomoExprTypes>(m, "PyomoExprTypes").def(py::init<>());
  py::class_<ex>(m, "ex");
  py::enum_<ExprType>(m, "ExprType")
      .value("py_float", ExprType::py_float)
      .value("var", ExprType::var)
      .value("param", ExprType::param)
      .value("product", ExprType::product)
      .value("sum", ExprType::sum)
      .value("negation", ExprType::negation)
      .value("external_func", ExprType::external_func)
      .value("power", ExprType::power)
      .value("division", ExprType::division)
      .value("unary_func", ExprType::unary_func)
      .value("linear", ExprType::linear)
      .value("named_expr", ExprType::named_expr)
      .value("numeric_constant", ExprType::numeric_constant)
      .export_values();
}
