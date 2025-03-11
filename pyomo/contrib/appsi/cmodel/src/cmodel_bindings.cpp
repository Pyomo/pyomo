/**___________________________________________________________________________
 *
 *  Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
 **/

#include "expression.hpp"
#include "fbbt_model.hpp"
#include "interval.hpp"
#include "lp_writer.hpp"
#include "model_base.hpp"
#include "nl_writer.hpp"
//#include "profiler.h"

extern double inf;

PYBIND11_MODULE(appsi_cmodel, m) {
  inf = py::module_::import("math").attr("inf").cast<double>();
  m.attr("inf") = inf;
  // m.def("ProfilerStart", &ProfilerStart);
  // m.def("ProfilerStop", &ProfilerStop);
  py::register_exception<IntervalException>(
      m, "IntervalException",
      py::module_::import("pyomo.common.errors").attr("IntervalException"));
  py::register_exception<InfeasibleConstraintException>(
      m, "InfeasibleConstraintException",
      py::module_::import("pyomo.common.errors")
          .attr("InfeasibleConstraintException"));
  m.def("_pow_with_inf", &_pow_with_inf);
  m.def("py_interval_add", &py_interval_add);
  m.def("py_interval_sub", &py_interval_sub);
  m.def("py_interval_mul", &py_interval_mul);
  m.def("py_interval_inv", &py_interval_inv);
  m.def("py_interval_div", &py_interval_div);
  m.def("py_interval_power", &py_interval_power);
  m.def("py_interval_exp", &py_interval_exp);
  m.def("py_interval_log", &py_interval_log);
  m.def("py_interval_abs", &py_interval_abs);
  m.def("_py_inverse_abs", &_py_inverse_abs);
  m.def("py_interval_log10", &py_interval_log10);
  m.def("py_interval_sin", &py_interval_sin);
  m.def("py_interval_cos", &py_interval_cos);
  m.def("py_interval_tan", &py_interval_tan);
  m.def("py_interval_asin", &py_interval_asin);
  m.def("py_interval_acos", &py_interval_acos);
  m.def("py_interval_atan", &py_interval_atan);
  m.def("_py_inverse_power1", &_py_inverse_power1);
  m.def("_py_inverse_power2", &_py_inverse_power2);
  m.def("process_lp_constraints", &process_lp_constraints);
  m.def("process_lp_objective", &process_lp_objective);
  m.def("process_nl_constraints", &process_nl_constraints);
  m.def("process_fbbt_constraints", &process_fbbt_constraints);
  m.def("process_pyomo_vars", &process_pyomo_vars);
  m.def("create_vars", &create_vars);
  m.def("create_params", &create_params);
  m.def("create_constants", &create_constants);
  m.def("appsi_exprs_from_pyomo_exprs", &appsi_exprs_from_pyomo_exprs);
  m.def("appsi_expr_from_pyomo_expr", &appsi_expr_from_pyomo_expr);
  m.def("prep_for_repn", &prep_for_repn);
  py::class_<PyomoExprTypes>(m, "PyomoExprTypes", py::module_local())
      .def(py::init<>());
  py::class_<Node, std::shared_ptr<Node>>(m, "Node")
      .def("is_variable_type", &Node::is_variable_type)
      .def("is_param_type", &Node::is_param_type)
      .def("is_expression_type", &Node::is_expression_type)
      .def("is_operator_type", &Node::is_operator_type)
      .def("is_constant_type", &Node::is_constant_type)
      .def("is_leaf", &Node::is_leaf);
  py::class_<ExpressionBase, Node, std::shared_ptr<ExpressionBase>>(
      m, "ExpressionBase")
      .def("__str__", &ExpressionBase::__str__)
      .def("evaluate", &ExpressionBase::evaluate);
  py::class_<Var, ExpressionBase, std::shared_ptr<Var>>(m, "Var")
      .def(py::init<>())
      .def(py::init<double>())
      .def(py::init<std::string, double>())
      .def(py::init<std::string>())
      .def("get_lb", &Var::get_lb)
      .def("get_ub", &Var::get_ub)
      .def("get_domain", &Var::get_domain)
      .def_readwrite("lb", &Var::lb)
      .def_readwrite("ub", &Var::ub)
      .def_readwrite("value", &Var::value)
      .def_readwrite("name", &Var::name)
      .def_readwrite("domain", &Var::domain)
      .def_readwrite("fixed", &Var::fixed);
  py::class_<Param, ExpressionBase, std::shared_ptr<Param>>(m, "Param")
      .def(py::init<>())
      .def(py::init<double>())
      .def(py::init<std::string, double>())
      .def(py::init<std::string>())
      .def_readwrite("name", &Param::name)
      .def_readwrite("value", &Param::value);
  py::class_<Constant, ExpressionBase, std::shared_ptr<Constant>>(m, "Constant")
      .def(py::init<>())
      .def(py::init<double>())
      .def_readwrite("value", &Constant::value);
  py::class_<Operator, Node, std::shared_ptr<Operator>>(m, "Operator");
  py::class_<Expression, ExpressionBase, std::shared_ptr<Expression>>(
      m, "Expression")
      .def(py::init<int>())
      .def("get_operators", &Expression::get_operators);
  py::class_<Objective, std::shared_ptr<Objective>>(m, "Objective")
      .def_readwrite("sense", &Objective::sense)
      .def_readwrite("name", &Objective::name)
      .def(py::init<>());
  py::class_<Constraint, std::shared_ptr<Constraint>>(m, "Constraint")
      .def_readwrite("lb", &Constraint::lb)
      .def_readwrite("ub", &Constraint::ub)
      .def_readwrite("active", &Constraint::active)
      .def_readwrite("name", &Constraint::name)
      .def(py::init<>());
  py::class_<Model>(m, "Model")
      .def_readwrite("constraints", &Model::constraints)
      .def_readwrite("objective", &Model::objective)
      .def("add_constraint", &Model::add_constraint)
      .def("remove_constraint", &Model::remove_constraint)
      .def(py::init<>());
  py::class_<FBBTObjective, Objective, std::shared_ptr<FBBTObjective>>(
      m, "FBBTObjective")
      .def_readwrite("expr", &FBBTObjective::expr)
      .def(py::init<std::shared_ptr<ExpressionBase>>());
  py::class_<FBBTConstraint, Constraint, std::shared_ptr<FBBTConstraint>>(
      m, "FBBTConstraint")
      .def_readwrite("body", &FBBTConstraint::body)
      .def("perform_fbbt", &FBBTConstraint::perform_fbbt)
      .def(py::init<std::shared_ptr<ExpressionBase>,
                    std::shared_ptr<ExpressionBase>,
                    std::shared_ptr<ExpressionBase>>());
  py::class_<FBBTModel, Model>(m, "FBBTModel")
      .def("perform_fbbt_with_seed", &FBBTModel::perform_fbbt_with_seed)
      .def("perform_fbbt", &FBBTModel::perform_fbbt)
      .def(py::init<>());
  py::class_<NLBase, std::shared_ptr<NLBase>>(m, "NLBase");
  py::class_<NLConstraint, NLBase, Constraint, std::shared_ptr<NLConstraint>>(
      m, "NLConstraint")
      .def(py::init<std::shared_ptr<ExpressionBase>,
                    std::vector<std::shared_ptr<ExpressionBase>>,
                    std::vector<std::shared_ptr<Var>>,
                    std::shared_ptr<ExpressionBase>>());
  py::class_<NLObjective, NLBase, Objective, std::shared_ptr<NLObjective>>(
      m, "NLObjective")
      .def(py::init<std::shared_ptr<ExpressionBase>,
                    std::vector<std::shared_ptr<ExpressionBase>>,
                    std::vector<std::shared_ptr<Var>>,
                    std::shared_ptr<ExpressionBase>>());
  py::class_<NLWriter, Model>(m, "NLWriter")
      .def(py::init<>())
      .def("write", &NLWriter::write)
      .def("get_solve_cons", &NLWriter::get_solve_cons)
      .def("get_solve_vars", &NLWriter::get_solve_vars);
  py::class_<LPBase, std::shared_ptr<LPBase>>(m, "LPBase");
  py::class_<LPConstraint, LPBase, Constraint, std::shared_ptr<LPConstraint>>(
      m, "LPConstraint")
      .def(py::init<>());
  py::class_<LPObjective, LPBase, Objective, std::shared_ptr<LPObjective>>(
      m, "LPObjective")
      .def(py::init<>());
  py::class_<LPWriter, Model>(m, "LPWriter")
      .def(py::init<>())
      .def("write", &LPWriter::write)
      .def("get_solve_cons", &LPWriter::get_solve_cons);
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
