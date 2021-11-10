/**___________________________________________________________________________
 *
 *  Pyomo: Python Optimization Modeling Objects
 * Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#include "lp_writer.hpp"
//#include "profiler.h"


PYBIND11_MODULE(appsi_cmodel, m)
{
  m.attr("inf") = inf;
  //m.def("ProfilerStart", &ProfilerStart);
  //m.def("ProfilerStop", &ProfilerStop);
  m.def("process_lp_constraints", &process_lp_constraints);
  m.def("create_vars", &create_vars);
  m.def("create_params", &create_params);
  m.def("create_constants", &create_constants);
  m.def("appsi_exprs_from_pyomo_exprs", &appsi_exprs_from_pyomo_exprs);
  m.def("appsi_expr_from_pyomo_expr", &appsi_expr_from_pyomo_expr);
  py::class_<PyomoExprTypes>(m, "PyomoExprTypes")
    .def(py::init<>());
  py::class_<Node, std::shared_ptr<Node> >(m, "Node")
    .def("is_variable_type", &Node::is_variable_type)
    .def("is_param_type", &Node::is_param_type)
    .def("is_expression_type", &Node::is_expression_type)
    .def("is_operator_type", &Node::is_operator_type)
    .def("is_constant_type", &Node::is_constant_type)
    .def("is_leaf", &Node::is_leaf);
  py::class_<ExpressionBase, Node, std::shared_ptr<ExpressionBase> >(m, "ExpressionBase")
    .def("__str__", &ExpressionBase::__str__)
    .def("evaluate", &ExpressionBase::evaluate);
  py::class_<Var, ExpressionBase, std::shared_ptr<Var> >(m, "Var")
    .def(py::init<>())
    .def(py::init<double>())
    .def(py::init<std::string, double>())
    .def(py::init<std::string>())
    .def_readwrite("lb", &Var::lb)
    .def_readwrite("ub", &Var::ub)
    .def_readwrite("value", &Var::value)
    .def_readwrite("name", &Var::name)
    .def_readwrite("domain", &Var::domain)
    .def_readwrite("fixed", &Var::fixed);
  py::class_<Param, ExpressionBase, std::shared_ptr<Param> >(m, "Param")
    .def(py::init<>())
    .def(py::init<double>())
    .def(py::init<std::string, double>())
    .def(py::init<std::string>())
    .def_readwrite("name", &Param::name)
    .def_readwrite("value", &Param::value);
  py::class_<Constant, ExpressionBase, std::shared_ptr<Constant> >(m, "Constant")
    .def(py::init<>())
    .def(py::init<double>())
    .def_readwrite("value", &Constant::value);
  py::class_<Operator, Node, std::shared_ptr<Operator> >(m, "Operator");
  py::class_<Expression, ExpressionBase, std::shared_ptr<Expression> >(m, "Expression")
    .def(py::init<int>())
    .def("get_operators", &Expression::get_operators);
  py::class_<NLBase, std::shared_ptr<NLBase> >(m, "NLBase");
  py::class_<NLConstraint, NLBase, std::shared_ptr<NLConstraint> >(m, "NLConstraint")
    .def_readwrite("lb", &NLConstraint::lb)
    .def_readwrite("ub", &NLConstraint::ub)
    .def_readwrite("active", &NLConstraint::active)
    .def(py::init<std::shared_ptr<ExpressionBase>,
	 std::vector<std::shared_ptr<ExpressionBase> >,
	 std::vector<std::shared_ptr<Var> >,
	 std::shared_ptr<ExpressionBase> >());
  py::class_<NLObjective, NLBase, std::shared_ptr<NLObjective> >(m, "NLObjective")
    .def_readwrite("sense", &NLObjective::sense)
    .def(py::init<std::shared_ptr<ExpressionBase>,
	 std::vector<std::shared_ptr<ExpressionBase> >,
	 std::vector<std::shared_ptr<Var> >,
	 std::shared_ptr<ExpressionBase> >());
  py::class_<NLWriter>(m, "NLWriter")
    .def(py::init<>())
    .def("write", &NLWriter::write)
    .def("add_constraint", &NLWriter::add_constraint)
    .def("remove_constraint", &NLWriter::remove_constraint)
    .def("get_solve_cons", &NLWriter::get_solve_cons)
    .def("get_solve_vars", &NLWriter::get_solve_vars)
    .def_readwrite("objective", &NLWriter::objective);
  py::class_<LPBase, std::shared_ptr<LPBase> >(m, "LPBase")
    .def_readwrite("name", &LPBase::name);    
  py::class_<LPConstraint, LPBase, std::shared_ptr<LPConstraint> >(m, "LPConstraint")
    .def_readwrite("lb", &LPConstraint::lb)
    .def_readwrite("ub", &LPConstraint::ub)
    .def_readwrite("active", &LPConstraint::active)
    .def(py::init<std::shared_ptr<ExpressionBase>,
	 std::vector<std::shared_ptr<ExpressionBase> >,
	 std::vector<std::shared_ptr<Var> >,
	 std::vector<std::shared_ptr<ExpressionBase> >,
	 std::vector<std::shared_ptr<Var> >,
	 std::vector<std::shared_ptr<Var> > >());
  py::class_<LPObjective, LPBase, std::shared_ptr<LPObjective> >(m, "LPObjective")
    .def_readwrite("sense", &LPObjective::sense)
    .def(py::init<std::shared_ptr<ExpressionBase>,
	 std::vector<std::shared_ptr<ExpressionBase> >,
	 std::vector<std::shared_ptr<Var> >,
	 std::vector<std::shared_ptr<ExpressionBase> >,
	 std::vector<std::shared_ptr<Var> >,
	 std::vector<std::shared_ptr<Var> > >());
  py::class_<LPWriter>(m, "LPWriter")
    .def(py::init<>())
    .def("write", &LPWriter::write)
    .def("add_constraint", &LPWriter::add_constraint)
    .def("remove_constraint", &LPWriter::remove_constraint)
    .def("get_solve_cons", &LPWriter::get_solve_cons)
    .def_readwrite("objective", &LPWriter::objective);
}
