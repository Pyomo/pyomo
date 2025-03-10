//  ___________________________________________________________________________
//
//  Pyomo: Python Optimization Modeling Objects
//  Copyright (c) 2008-2025
//  National Technology and Engineering Solutions of Sandia, LLC
//  Under the terms of Contract DE-NA0003525 with National Technology and
//  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
//  rights in this software.
//  This software is distributed under the 3-clause BSD License.
//  ___________________________________________________________________________

#include "ginac_interface.hpp"


bool is_integer(double x) {
  return std::floor(x) == x;
}


ex ginac_expr_from_pyomo_node(
  py::handle expr, 
  std::unordered_map<long, ex> &leaf_map, 
  std::unordered_map<ex, py::object> &ginac_pyomo_map, 
  PyomoExprTypes &expr_types,
  bool symbolic_solver_labels
  ) {
  ex res;
  ExprType tmp_type =
      expr_types.expr_type_map[py::type::of(expr)].cast<ExprType>();

  switch (tmp_type) {
  case py_float: {
    double val = expr.cast<double>();
    if (is_integer(val)) {
      res = numeric((long) val);
    }
    else {
      res = numeric(val);
    }
    break;
  }
  case var: {
    long expr_id = expr_types.id(expr).cast<long>();
    if (leaf_map.count(expr_id) == 0) {
      std::string vname;
      if (symbolic_solver_labels) {
        vname = expr.attr("name").cast<std::string>();
      }
      else {
        vname = "x" + std::to_string(expr_id);
      }
      py::object lb = expr.attr("lb");
      if (lb.is_none() || lb.cast<double>() < 0) {
        leaf_map[expr_id] = realsymbol(vname);
      }
      else {
        leaf_map[expr_id] = possymbol(vname);
      }
      ginac_pyomo_map[leaf_map[expr_id]] = expr.cast<py::object>();
    }
    res = leaf_map[expr_id];
    break;
  }
  case param: {
    long expr_id = expr_types.id(expr).cast<long>();
    if (leaf_map.count(expr_id) == 0) {
      std::string pname;
      if (symbolic_solver_labels) {
        pname = expr.attr("name").cast<std::string>();
      }
      else {
        pname = "p" + std::to_string(expr_id);
      }
      leaf_map[expr_id] = realsymbol(pname);
      ginac_pyomo_map[leaf_map[expr_id]] = expr.cast<py::object>();
    }
    res = leaf_map[expr_id];
    break;
  }
  case product: {
    py::list pyomo_args = expr.attr("args");
    res = ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels) * ginac_expr_from_pyomo_node(pyomo_args[1], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
    break;
  }
  case sum: {
    py::list pyomo_args = expr.attr("args");
    for (py::handle arg : pyomo_args) {
      res += ginac_expr_from_pyomo_node(arg, leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
    }
    break;
  }
  case negation: {
    py::list pyomo_args = expr.attr("args");
    res = - ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
    break;
  }
  case external_func: {
    long expr_id = expr_types.id(expr).cast<long>();
    if (leaf_map.count(expr_id) == 0) {
      leaf_map[expr_id] = realsymbol("f" + std::to_string(expr_id));
      ginac_pyomo_map[leaf_map[expr_id]] = expr.cast<py::object>();
    }
    res = leaf_map[expr_id];
    break;
  }
  case ExprType::power: {
    py::list pyomo_args = expr.attr("args");
    res = pow(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels), ginac_expr_from_pyomo_node(pyomo_args[1], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    break;
  }
  case division: {
    py::list pyomo_args = expr.attr("args");
    res = ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels) / ginac_expr_from_pyomo_node(pyomo_args[1], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
    break;
  }
  case unary_func: {
    std::string function_name = expr.attr("getname")().cast<std::string>();
    py::list pyomo_args = expr.attr("args");
    if (function_name == "exp")
      res = exp(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "log")
      res = log(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "sin")
      res = sin(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "cos")
      res = cos(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "tan")
      res = tan(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "asin")
      res = asin(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "acos")
      res = acos(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "atan")
      res = atan(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else if (function_name == "sqrt")
      res = sqrt(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
    else
      throw py::value_error("Unrecognized expression type: " + function_name);
    break;
  }
  case linear: {
    py::list pyomo_args = expr.attr("args");
    for (py::handle arg : pyomo_args) {
      res += ginac_expr_from_pyomo_node(arg, leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
    }
    break;
  }
  case named_expr: {
    res = ginac_expr_from_pyomo_node(expr.attr("expr"), leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
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
    res = abs(ginac_expr_from_pyomo_node(pyomo_args[0], leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels));
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

ex pyomo_expr_to_ginac_expr(
  py::handle expr,
  std::unordered_map<long, ex> &leaf_map,
  std::unordered_map<ex, py::object> &ginac_pyomo_map,
  PyomoExprTypes &expr_types,
  bool symbolic_solver_labels
  ) {
    ex res = ginac_expr_from_pyomo_node(expr, leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
    return res;
  }

ex pyomo_to_ginac(py::handle expr, PyomoExprTypes &expr_types) {
  std::unordered_map<long, ex> leaf_map;
  std::unordered_map<ex, py::object> ginac_pyomo_map;
  ex res = ginac_expr_from_pyomo_node(expr, leaf_map, ginac_pyomo_map, expr_types, true);
  return res;
}


class GinacToPyomoVisitor 
: public visitor, 
  public symbol::visitor, 
  public numeric::visitor, 
  public add::visitor, 
  public mul::visitor, 
  public GiNaC::power::visitor, 
  public function::visitor, 
  public basic::visitor
{
  public:
  std::unordered_map<ex, py::object> *leaf_map;
  std::unordered_map<ex, py::object> node_map;
  PyomoExprTypes *expr_types;

  GinacToPyomoVisitor(std::unordered_map<ex, py::object> *_leaf_map, PyomoExprTypes *_expr_types) : leaf_map(_leaf_map), expr_types(_expr_types) {}
  ~GinacToPyomoVisitor() = default;

  void visit(const symbol& e) {
    node_map[e] = leaf_map->at(e);
  }

  void visit(const numeric& e) {
    double val = e.to_double();
    node_map[e] = expr_types->NumericConstant(py::cast(val));
  }

  void visit(const add& e) {
    size_t n = e.nops();
    py::object pe = node_map[e.op(0)];
    for (unsigned long ndx=1; ndx < n; ++ndx) {
      pe = pe.attr("__add__")(node_map[e.op(ndx)]);
    }
    node_map[e] = pe;
  }

  void visit(const mul& e) {
    size_t n = e.nops();
    py::object pe = node_map[e.op(0)];
    for (unsigned long ndx=1; ndx < n; ++ndx) {
      pe = pe.attr("__mul__")(node_map[e.op(ndx)]);
    }
    node_map[e] = pe;
  }

  void visit(const GiNaC::power& e) {
    py::object arg1 = node_map[e.op(0)];
    py::object arg2 = node_map[e.op(1)];
    py::object pe = arg1.attr("__pow__")(arg2);
    node_map[e] = pe;
  }

  void visit(const function& e) {
    py::object arg = node_map[e.op(0)];
    std::string func_type = e.get_name();
    py::object pe;
    if (func_type == "exp") {
      pe = expr_types->exp(arg);
    }
    else if (func_type == "log") {
      pe = expr_types->log(arg);
    }
    else if (func_type == "sin") {
      pe = expr_types->sin(arg);
    }
    else if (func_type == "cos") {
      pe = expr_types->cos(arg);
    }
    else if (func_type == "tan") {
      pe = expr_types->tan(arg);
    }
    else if (func_type == "asin") {
      pe = expr_types->asin(arg);
    }
    else if (func_type == "acos") {
      pe = expr_types->acos(arg);
    }
    else if (func_type == "atan") {
      pe = expr_types->atan(arg);
    }
    else if (func_type == "sqrt") {
      pe = expr_types->sqrt(arg);
    }
    else {
      throw py::value_error("unrecognized unary function: " + func_type);
    }
    node_map[e] = pe;
  }

  void visit(const basic& e) {
    throw py::value_error("unrecognized ginac expression type");
  }
};


ex GinacInterface::to_ginac(py::handle expr) {
  return pyomo_expr_to_ginac_expr(expr, leaf_map, ginac_pyomo_map, expr_types, symbolic_solver_labels);
}

py::object GinacInterface::from_ginac(ex &ge) {
  GinacToPyomoVisitor v(&ginac_pyomo_map, &expr_types);
  ge.traverse_postorder(v);
  return v.node_map[ge];
}

PYBIND11_MODULE(ginac_interface, m) {
  m.def("pyomo_to_ginac", &pyomo_to_ginac);
  py::class_<PyomoExprTypes>(m, "PyomoExprTypes", py::module_local())
    .def(py::init<>());
  py::class_<ex>(m, "ginac_expression")
    .def("expand", [](ex &ge) {
      return ge.expand();
    })
    .def("normal", &ex::normal)
    .def("__str__", [](ex &ge) {
	std::ostringstream stream;
	stream << ge;
	return stream.str();
      });
  py::class_<GinacInterface>(m, "GinacInterface")
    .def(py::init<bool>())
    .def("to_ginac", &GinacInterface::to_ginac)
    .def("from_ginac", &GinacInterface::from_ginac);
  py::enum_<ExprType>(m, "ExprType", py::module_local())
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
