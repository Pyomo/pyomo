/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 * National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#include "lp_writer.hpp"

void write_expr(std::ofstream &f, std::shared_ptr<LPBase> obj,
                bool is_objective) {
  double coef;
  for (unsigned int ndx = 0; ndx < obj->linear_coefficients->size(); ++ndx) {
    coef = obj->linear_coefficients->at(ndx)->evaluate();
    if (coef >= 0) {
      f << "+";
    } else {
      f << "-";
    }
    f << std::abs(coef) << " ";
    f << obj->linear_vars->at(ndx)->name << " \n";
  }
  if (is_objective) {
    f << "+1 obj_const \n";
  }

  if (obj->quadratic_coefficients->size() != 0) {
    f << "+ [ \n";
    for (unsigned int ndx = 0; ndx < obj->quadratic_coefficients->size();
         ++ndx) {
      coef = obj->quadratic_coefficients->at(ndx)->evaluate();
      if (is_objective) {
        coef *= 2;
      }
      if (coef >= 0) {
        f << "+";
      } else {
        f << "-";
      }
      f << std::abs(coef) << " ";
      f << obj->quadratic_vars_1->at(ndx)->name << " * ";
      f << obj->quadratic_vars_2->at(ndx)->name << " \n";
    }
    f << "] ";
    if (is_objective) {
      f << "/ 2 ";
    }
    f << "\n";
  }
}

void LPWriter::write(std::string filename) {
  std::ofstream f;
  f.open(filename);
  f.precision(17);

  std::shared_ptr<LPObjective> lp_objective =
      std::dynamic_pointer_cast<LPObjective>(objective);

  if (lp_objective->sense == 0) {
    f << "minimize\n";
  } else {
    f << "maximize\n";
  }

  f << lp_objective->name << ": \n";
  write_expr(f, lp_objective, true);

  f << "\ns.t.\n\n";

  std::vector<std::shared_ptr<LPConstraint>> sorted_constraints;
  for (std::shared_ptr<Constraint> con : constraints) {
    sorted_constraints.push_back(std::dynamic_pointer_cast<LPConstraint>(con));
  }
  int sorted_con_index = 0;
  for (std::shared_ptr<LPConstraint> con : sorted_constraints) {
    con->index = sorted_con_index;
    sorted_con_index += 1;
  }
  current_con_ndx = constraints.size();

  std::vector<std::shared_ptr<LPConstraint>> active_constraints;
  for (std::shared_ptr<LPConstraint> con : sorted_constraints) {
    if (con->active) {
      active_constraints.push_back(con);
    }
  }

  double con_lb;
  double con_ub;
  double body_constant_val;
  for (std::shared_ptr<LPConstraint> con : active_constraints) {
    con_lb = con->lb->evaluate();
    con_ub = con->ub->evaluate();
    body_constant_val = con->constant_expr->evaluate();
    if (con_lb == con_ub) {
      con_lb -= body_constant_val;
      con_ub = con_lb;
      f << con->name << "_eq: \n";
      write_expr(f, con, false);
      f << "= " << con_lb << " \n\n";
    } else if (con_lb > -inf && con_ub < inf) {
      con_lb -= body_constant_val;
      con_ub -= body_constant_val;
      f << con->name << "_lb: \n";
      write_expr(f, con, false);
      f << ">= " << con_lb << " \n\n";
      f << con->name << "_ub: \n";
      write_expr(f, con, false);
      f << "<= " << con_ub << " \n\n";
    } else if (con_lb > -inf) {
      con_lb -= body_constant_val;
      f << con->name << "_lb: \n";
      write_expr(f, con, false);
      f << ">= " << con_lb << " \n\n";
    } else if (con_ub < inf) {
      con_ub -= body_constant_val;
      f << con->name << "_ub: \n";
      write_expr(f, con, false);
      f << "<= " << con_ub << " \n\n";
    }
  }

  f << "obj_const_con_eq: \n";
  f << "+1 obj_const \n";
  f << "= " << lp_objective->constant_expr->evaluate() << " \n\n";

  for (std::shared_ptr<LPConstraint> con : active_constraints) {
    for (std::shared_ptr<Var> v : *(con->linear_vars)) {
      v->index = -1;
    }
    for (std::shared_ptr<Var> v : *(con->quadratic_vars_1)) {
      v->index = -1;
    }
    for (std::shared_ptr<Var> v : *(con->quadratic_vars_2)) {
      v->index = -1;
    }
  }

  for (std::shared_ptr<Var> v : *(lp_objective->linear_vars)) {
    v->index = -1;
  }
  for (std::shared_ptr<Var> v : *(lp_objective->quadratic_vars_1)) {
    v->index = -1;
  }
  for (std::shared_ptr<Var> v : *(lp_objective->quadratic_vars_2)) {
    v->index = -1;
  }

  std::vector<std::shared_ptr<Var>> active_vars;
  for (std::shared_ptr<LPConstraint> con : active_constraints) {
    for (std::shared_ptr<Var> v : *(con->linear_vars)) {
      if (v->index == -1) {
        v->index = -2;
        active_vars.push_back(v);
      }
    }
    for (std::shared_ptr<Var> v : *(con->quadratic_vars_1)) {
      if (v->index == -1) {
        v->index = -2;
        active_vars.push_back(v);
      }
    }
    for (std::shared_ptr<Var> v : *(con->quadratic_vars_2)) {
      if (v->index == -1) {
        v->index = -2;
        active_vars.push_back(v);
      }
    }
  }

  for (std::shared_ptr<Var> v : *(lp_objective->linear_vars)) {
    if (v->index == -1) {
      v->index = -2;
      active_vars.push_back(v);
    }
  }
  for (std::shared_ptr<Var> v : *(lp_objective->quadratic_vars_1)) {
    if (v->index == -1) {
      v->index = -2;
      active_vars.push_back(v);
    }
  }
  for (std::shared_ptr<Var> v : *(lp_objective->quadratic_vars_2)) {
    if (v->index == -1) {
      v->index = -2;
      active_vars.push_back(v);
    }
  }

  f << "Bounds\n";
  std::vector<std::shared_ptr<Var>> binaries;
  std::vector<std::shared_ptr<Var>> integer_vars;
  double v_lb;
  double v_ub;
  Domain v_domain;
  for (std::shared_ptr<Var> v : active_vars) {
    v_domain = v->get_domain();
    if (v_domain == binary) {
      binaries.push_back(v);
    } else if (v_domain == integers) {
      integer_vars.push_back(v);
    }
    if (v->fixed) {
      f << "  " << v->value << " <= " << v->name << " <= " << v->value << " \n";
    } else {
      v_lb = v->get_lb();
      v_ub = v->get_ub();
      f << "  ";
      if (v_lb <= -inf) {
        f << "-inf";
      } else {
        f << v_lb;
      }
      f << " <= " << v->name << " <= ";
      if (v_ub >= inf) {
        f << "+inf";
      } else {
        f << v_ub;
      }
      f << " \n";
    }
  }
  f << "-inf <= obj_const <= +inf\n\n";

  if (binaries.size() > 0) {
    f << "Binaries \n";
    for (std::shared_ptr<Var> v : binaries) {
      f << v->name << " \n";
    }
  }

  if (integer_vars.size() > 0) {
    f << "Generals \n";
    for (std::shared_ptr<Var> v : integer_vars) {
      f << v->name << " \n";
    }
  }

  f << "end\n";

  f.close();

  solve_cons = active_constraints;
  solve_vars = active_vars;
}

std::vector<std::shared_ptr<Var>> LPWriter::get_solve_vars() {
  return solve_vars;
}

std::vector<std::shared_ptr<LPConstraint>> LPWriter::get_solve_cons() {
  return solve_cons;
}

void process_lp_constraints(py::list cons, py::object writer) {
  py::object generate_standard_repn =
      py::module_::import("pyomo.repn.standard_repn")
          .attr("generate_standard_repn");
  py::object id = py::module_::import("builtins").attr("id");
  py::str cname;
  py::object repn;
  py::object getSymbol = writer.attr("_symbol_map").attr("getSymbol");
  py::object labeler = writer.attr("_con_labeler");
  LPWriter *c_writer = writer.attr("_writer").cast<LPWriter *>();
  py::dict var_map = writer.attr("_pyomo_var_to_solver_var_map");
  py::dict param_map = writer.attr("_pyomo_param_to_solver_param_map");
  py::dict pyomo_con_to_solver_con_map =
      writer.attr("_pyomo_con_to_solver_con_map");
  py::dict solver_con_to_pyomo_con_map =
      writer.attr("_solver_con_to_pyomo_con_map");
  std::shared_ptr<ExpressionBase> _const;
  py::object repn_constant;
  py::list repn_linear_coefs;
  py::list repn_linear_vars;
  py::list repn_quad_coefs;
  py::list repn_quad_vars;
  std::shared_ptr<LPConstraint> lp_con;
  py::tuple v_tuple;
  py::handle lb;
  py::handle ub;
  py::tuple lower_body_upper;
  py::dict active_constraints = writer.attr("_active_constraints");
  py::object nonlinear_expr;
  PyomoExprTypes expr_types = PyomoExprTypes();
  for (py::handle c : cons) {
    lower_body_upper = c.attr("to_bounded_expression")();
    cname = getSymbol(c, labeler);
    repn = generate_standard_repn(
        lower_body_upper[1], "compute_values"_a = false, "quadratic"_a = true);
    nonlinear_expr = repn.attr("nonlinear_expr");
    if (!(nonlinear_expr.is(py::none()))) {
      throw py::value_error(
          "cannot write an LP file with a nonlinear constraint");
    }
    repn_constant = repn.attr("constant");
    _const = appsi_expr_from_pyomo_expr(repn_constant, var_map, param_map,
                                        expr_types);
    std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>> lin_coef =
        std::make_shared<std::vector<std::shared_ptr<ExpressionBase>>>();
    std::shared_ptr<std::vector<std::shared_ptr<Var>>> lin_vars =
        std::make_shared<std::vector<std::shared_ptr<Var>>>();
    ;
    std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>> quad_coef =
        std::make_shared<std::vector<std::shared_ptr<ExpressionBase>>>();
    ;
    std::shared_ptr<std::vector<std::shared_ptr<Var>>> quad_vars_1 =
        std::make_shared<std::vector<std::shared_ptr<Var>>>();
    ;
    std::shared_ptr<std::vector<std::shared_ptr<Var>>> quad_vars_2 =
        std::make_shared<std::vector<std::shared_ptr<Var>>>();
    ;
    repn_linear_coefs = repn.attr("linear_coefs");
    for (py::handle coef : repn_linear_coefs) {
      lin_coef->push_back(
          appsi_expr_from_pyomo_expr(coef, var_map, param_map, expr_types));
    }
    repn_linear_vars = repn.attr("linear_vars");
    for (py::handle v : repn_linear_vars) {
      lin_vars->push_back(var_map[id(v)].cast<std::shared_ptr<Var>>());
    }
    repn_quad_coefs = repn.attr("quadratic_coefs");
    for (py::handle coef : repn_quad_coefs) {
      quad_coef->push_back(
          appsi_expr_from_pyomo_expr(coef, var_map, param_map, expr_types));
    }
    repn_quad_vars = repn.attr("quadratic_vars");
    for (py::handle v_tuple_handle : repn_quad_vars) {
      v_tuple = v_tuple_handle.cast<py::tuple>();
      quad_vars_1->push_back(
          var_map[id(v_tuple[0])].cast<std::shared_ptr<Var>>());
      quad_vars_2->push_back(
          var_map[id(v_tuple[1])].cast<std::shared_ptr<Var>>());
    }

    lp_con = std::make_shared<LPConstraint>();
    lp_con->name = cname;
    lp_con->constant_expr = _const;
    lp_con->linear_coefficients = lin_coef;
    lp_con->linear_vars = lin_vars;
    lp_con->quadratic_coefficients = quad_coef;
    lp_con->quadratic_vars_1 = quad_vars_1;
    lp_con->quadratic_vars_2 = quad_vars_2;

    lb = lower_body_upper[0];
    ub = lower_body_upper[2];
    if (!lb.is(py::none())) {
      lp_con->lb =
          appsi_expr_from_pyomo_expr(lb, var_map, param_map, expr_types);
    }
    if (!ub.is(py::none())) {
      lp_con->ub =
          appsi_expr_from_pyomo_expr(ub, var_map, param_map, expr_types);
    }
    c_writer->add_constraint(lp_con);
    pyomo_con_to_solver_con_map[c] = py::cast(lp_con);
    solver_con_to_pyomo_con_map[py::cast(lp_con)] = c;
  }
}

std::shared_ptr<LPObjective> process_lp_objective(PyomoExprTypes &expr_types,
                                                  py::object pyomo_obj,
                                                  py::dict var_map,
                                                  py::dict param_map) {
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>> lin_coef =
      std::make_shared<std::vector<std::shared_ptr<ExpressionBase>>>();
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> lin_vars =
      std::make_shared<std::vector<std::shared_ptr<Var>>>();
  ;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>> quad_coef =
      std::make_shared<std::vector<std::shared_ptr<ExpressionBase>>>();
  ;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> quad_vars_1 =
      std::make_shared<std::vector<std::shared_ptr<Var>>>();
  ;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> quad_vars_2 =
      std::make_shared<std::vector<std::shared_ptr<Var>>>();
  ;
  std::shared_ptr<ExpressionBase> _const;

  if (pyomo_obj.is(py::none())) {
    _const = std::make_shared<Constant>(0);
  } else {
    py::object generate_standard_repn =
        py::module_::import("pyomo.repn.standard_repn")
            .attr("generate_standard_repn");
    py::object repn = generate_standard_repn(pyomo_obj.attr("expr"),
                                             "compute_values"_a = false,
                                             "quadratic"_a = true);
    _const = appsi_expr_from_pyomo_expr(repn.attr("constant"), var_map,
                                        param_map, expr_types);
    py::handle repn_linear_coefs = repn.attr("linear_coefs");
    for (py::handle coef : repn_linear_coefs) {
      lin_coef->push_back(
          appsi_expr_from_pyomo_expr(coef, var_map, param_map, expr_types));
    }
    py::handle repn_linear_vars = repn.attr("linear_vars");
    for (py::handle v : repn_linear_vars) {
      lin_vars->push_back(
          var_map[expr_types.id(v)].cast<std::shared_ptr<Var>>());
    }
    py::handle repn_quad_coefs = repn.attr("quadratic_coefs");
    for (py::handle coef : repn_quad_coefs) {
      quad_coef->push_back(
          appsi_expr_from_pyomo_expr(coef, var_map, param_map, expr_types));
    }
    py::handle repn_quad_vars = repn.attr("quadratic_vars");
    py::tuple v_tuple;
    for (py::handle v_tuple_handle : repn_quad_vars) {
      v_tuple = v_tuple_handle.cast<py::tuple>();
      quad_vars_1->push_back(
          var_map[expr_types.id(v_tuple[0])].cast<std::shared_ptr<Var>>());
      quad_vars_2->push_back(
          var_map[expr_types.id(v_tuple[1])].cast<std::shared_ptr<Var>>());
    }
  }

  std::shared_ptr<LPObjective> lp_obj = std::make_shared<LPObjective>();
  lp_obj->constant_expr = _const;
  lp_obj->linear_coefficients = lin_coef;
  lp_obj->linear_vars = lin_vars;
  lp_obj->quadratic_coefficients = quad_coef;
  lp_obj->quadratic_vars_1 = quad_vars_1;
  lp_obj->quadratic_vars_2 = quad_vars_2;

  return lp_obj;
}
