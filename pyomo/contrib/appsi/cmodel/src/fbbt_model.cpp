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

#include "fbbt_model.hpp"

FBBTObjective::FBBTObjective(std::shared_ptr<ExpressionBase> _expr)
    : Objective() {
  expr = _expr;
}

FBBTConstraint::FBBTConstraint(std::shared_ptr<ExpressionBase> _lb,
                               std::shared_ptr<ExpressionBase> _body,
                               std::shared_ptr<ExpressionBase> _ub)
    : Constraint() {
  lb = _lb;
  body = _body;
  ub = _ub;
  variables = body->identify_variables();

  if (body->is_expression_type()) {
    std::shared_ptr<Expression> e = std::dynamic_pointer_cast<Expression>(body);
    lbs = new double[e->n_operators];
    ubs = new double[e->n_operators];
  } else {
    lbs = new double[1];
    ubs = new double[1];
  }
}

FBBTConstraint::~FBBTConstraint() {
  delete[] lbs;
  delete[] ubs;
}

void FBBTConstraint::perform_fbbt(double feasibility_tol, double integer_tol,
                                  double improvement_tol,
                                  std::set<std::shared_ptr<Var>> &improved_vars,
                                  bool deactivate_satisfied_constraints) {
  double body_lb;
  double body_ub;

  if (body->is_expression_type()) {
    std::shared_ptr<Expression> e = std::dynamic_pointer_cast<Expression>(body);
    e->propagate_bounds_forward(lbs, ubs, feasibility_tol, integer_tol);
  }

  body_lb = body->get_lb_from_array(lbs);
  body_ub = body->get_ub_from_array(ubs);

  double con_lb = lb->evaluate();
  double con_ub = ub->evaluate();

  if (body_lb > con_ub + feasibility_tol ||
      body_ub < con_lb - feasibility_tol) {
    throw InfeasibleConstraintException(
        "Infeasible constraint (" + name + "); the bounds computed on the body of the "
        "constraint violate the constraint bounds:\n  con LB: " +
        std::to_string(con_lb) + "\n  con UB: " + std::to_string(con_ub) +
        "\n  body LB: " + std::to_string(body_lb) +
        "\n body UB: " + std::to_string(body_ub) + "\n");
  }

  if (deactivate_satisfied_constraints) {
    if (body_lb >= con_lb - feasibility_tol &&
        body_ub <= con_ub + feasibility_tol)
      active = false;
  }

  if (con_lb > body_lb ||
      con_ub < body_ub) // otherwise the constraint is always satisfied
  {
    if (con_lb > body_lb) {
      body_lb = con_lb;
    }
    if (con_ub < body_ub) {
      body_ub = con_ub;
    }
    body->set_bounds_in_array(body_lb, body_ub, lbs, ubs, feasibility_tol,
                              integer_tol, improvement_tol, improved_vars);
    if (body->is_expression_type()) {
      std::shared_ptr<Expression> e =
          std::dynamic_pointer_cast<Expression>(body);
      e->propagate_bounds_backward(lbs, ubs, feasibility_tol, integer_tol,
                                   improvement_tol, improved_vars);
    }
  }
}

std::shared_ptr<std::map<std::shared_ptr<Var>,
                         std::vector<std::shared_ptr<FBBTConstraint>>>>
FBBTModel::get_var_to_con_map() {
  std::shared_ptr<std::map<std::shared_ptr<Var>,
                           std::vector<std::shared_ptr<FBBTConstraint>>>>
      var_to_con_map = std::make_shared<
          std::map<std::shared_ptr<Var>,
                   std::vector<std::shared_ptr<FBBTConstraint>>>>();
  current_con_ndx = 0;
  std::shared_ptr<FBBTConstraint> fbbt_c;
  for (const std::shared_ptr<Constraint> &c : constraints) {
    c->index = current_con_ndx;
    current_con_ndx += 1;
    fbbt_c = std::dynamic_pointer_cast<FBBTConstraint>(c);
    for (const std::shared_ptr<Var> &v : *(fbbt_c->variables)) {
      (*var_to_con_map)[v].push_back(fbbt_c);
    }
  }
  return var_to_con_map;
}

unsigned int FBBTModel::perform_fbbt_on_cons(
    std::vector<std::shared_ptr<FBBTConstraint>> &seed_cons,
    double feasibility_tol, double integer_tol, double improvement_tol,
    int max_iter, bool deactivate_satisfied_constraints,
    std::shared_ptr<std::map<std::shared_ptr<Var>,
                             std::vector<std::shared_ptr<FBBTConstraint>>>>
        var_to_con_map) {
  std::set<std::shared_ptr<Var>> improved_vars_set;

  std::vector<std::shared_ptr<FBBTConstraint>> cons_to_fbbt = seed_cons;
  std::set<std::shared_ptr<FBBTConstraint>> cons_to_fbbt_set;
  unsigned int _iter = 0;
  while (_iter < max_iter * constraints.size() && cons_to_fbbt.size() > 0) {
    _iter += cons_to_fbbt.size();
    for (const std::shared_ptr<FBBTConstraint> &c : cons_to_fbbt) {
      c->perform_fbbt(feasibility_tol, integer_tol, improvement_tol,
                      improved_vars_set, deactivate_satisfied_constraints);
    }

    cons_to_fbbt.clear();
    cons_to_fbbt_set.clear();

    for (const std::shared_ptr<Var> &v : improved_vars_set) {
      for (const std::shared_ptr<FBBTConstraint> &c : var_to_con_map->at(v)) {
        if (cons_to_fbbt_set.count(c) == 0) {
          cons_to_fbbt_set.insert(c);
          cons_to_fbbt.push_back(c);
        }
      }
    }
    std::sort(cons_to_fbbt.begin(), cons_to_fbbt.end(), constraint_sorter);
    improved_vars_set.clear();
  }

  return _iter;
}

unsigned int
FBBTModel::perform_fbbt_with_seed(std::shared_ptr<Var> seed_var,
                                  double feasibility_tol, double integer_tol,
                                  double improvement_tol, int max_iter,
                                  bool deactivate_satisfied_constraints) {
  std::shared_ptr<std::map<std::shared_ptr<Var>,
                           std::vector<std::shared_ptr<FBBTConstraint>>>>
      var_to_con_map = get_var_to_con_map();

  std::vector<std::shared_ptr<FBBTConstraint>> &seed_cons =
      var_to_con_map->at(seed_var);

  return perform_fbbt_on_cons(seed_cons, feasibility_tol, integer_tol,
                              improvement_tol, max_iter,
                              deactivate_satisfied_constraints, var_to_con_map);
}

unsigned int FBBTModel::perform_fbbt(double feasibility_tol, double integer_tol,
                                     double improvement_tol, int max_iter,
                                     bool deactivate_satisfied_constraints) {
  std::shared_ptr<std::map<std::shared_ptr<Var>,
                           std::vector<std::shared_ptr<FBBTConstraint>>>>
      var_to_con_map = get_var_to_con_map();

  int n_cons = constraints.size();
  std::vector<std::shared_ptr<FBBTConstraint>> con_vector(n_cons);

  unsigned int ndx = 0;
  for (const std::shared_ptr<Constraint> &c : constraints) {
    con_vector[ndx] = std::dynamic_pointer_cast<FBBTConstraint>(c);
    ndx += 1;
  }

  return perform_fbbt_on_cons(con_vector, feasibility_tol, integer_tol,
                              improvement_tol, max_iter,
                              deactivate_satisfied_constraints, var_to_con_map);
}

void process_fbbt_constraints(FBBTModel *model, PyomoExprTypes &expr_types,
                              py::list cons, py::dict var_map,
                              py::dict param_map, py::dict active_constraints,
                              py::dict con_map, py::dict rev_con_map) {
  std::shared_ptr<FBBTConstraint> ccon;
  std::shared_ptr<ExpressionBase> ccon_lb;
  std::shared_ptr<ExpressionBase> ccon_ub;
  std::shared_ptr<ExpressionBase> ccon_body;
  py::tuple lower_body_upper;
  py::handle con_lb;
  py::handle con_ub;
  py::handle con_body;

  for (py::handle c : cons) {
    lower_body_upper = c.attr("to_bounded_expression")();
    con_lb = lower_body_upper[0];
    con_body = lower_body_upper[1];
    con_ub = lower_body_upper[2];

    ccon_body =
        appsi_expr_from_pyomo_expr(con_body, var_map, param_map, expr_types);

    if (con_lb.is(py::none())) {
      ccon_lb = std::make_shared<Constant>(-inf);
    } else {
      ccon_lb =
          appsi_expr_from_pyomo_expr(con_lb, var_map, param_map, expr_types);
    }

    if (con_ub.is(py::none())) {
      ccon_ub = std::make_shared<Constant>(inf);
    } else {
      ccon_ub =
          appsi_expr_from_pyomo_expr(con_ub, var_map, param_map, expr_types);
    }

    ccon = std::make_shared<FBBTConstraint>(ccon_lb, ccon_body, ccon_ub);
    model->add_constraint(ccon);
    con_map[c] = py::cast(ccon);
    rev_con_map[py::cast(ccon)] = c;
  }
}
