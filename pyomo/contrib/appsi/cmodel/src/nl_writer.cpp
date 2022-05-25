#include "nl_writer.hpp"

NLBase::NLBase(
    std::shared_ptr<ExpressionBase> _constant_expr,
    std::vector<std::shared_ptr<ExpressionBase>> &_linear_coefficients,
    std::vector<std::shared_ptr<Var>> &_linear_vars,
    std::shared_ptr<ExpressionBase> _nonlinear_expr) {
  constant_expr = _constant_expr;
  nonlinear_vars = _nonlinear_expr->identify_variables();

  external_operators = _nonlinear_expr->identify_external_operators();

  linear_vars = std::make_shared<std::vector<std::shared_ptr<Var>>>();
  for (std::shared_ptr<Var> v : _linear_vars) {
    linear_vars->push_back(v);
  }

  all_vars = std::make_shared<std::vector<std::shared_ptr<Var>>>();
  all_linear_coefficients =
      std::make_shared<std::vector<std::shared_ptr<ExpressionBase>>>();

  for (std::shared_ptr<Var> v : *nonlinear_vars) {
    v->index = -1;
  }

  std::shared_ptr<Var> v;
  for (unsigned int i = 0; i < linear_vars->size(); ++i) {
    v = linear_vars->at(i);
    v->index = i;
    all_vars->push_back(v);
    all_linear_coefficients->push_back(_linear_coefficients.at(i));
  }

  for (std::shared_ptr<Var> v : *nonlinear_vars) {
    if (v->index == -1) {
      all_vars->push_back(v);
      all_linear_coefficients->push_back(std::make_shared<Constant>(0));
    }
  }

  nonlinear_prefix_notation = _nonlinear_expr->get_prefix_notation();
}

bool NLBase::is_nonlinear() {
  if (nonlinear_prefix_notation->size() == 1) {
    std::shared_ptr<Node> node = nonlinear_prefix_notation->at(0);
    assert(node->is_constant_type());
    assert(std::dynamic_pointer_cast<Constant>(node)->evaluate() == 0);
    return false;
  } else {
    return true;
  }
}

bool variable_sorter(std::pair<std::shared_ptr<Var>, double> p1,
                     std::pair<std::shared_ptr<Var>, double> p2) {
  return p1.first->index < p2.first->index;
}

void NLWriter::write(std::string filename) {
  std::ofstream f;
  f.open(filename);
  f.precision(17);

  std::vector<std::shared_ptr<NLConstraint>> sorted_constraints;
  for (std::shared_ptr<Constraint> con : constraints) {
    sorted_constraints.push_back(std::dynamic_pointer_cast<NLConstraint>(con));
  }
  int sorted_con_index = 0;
  for (std::shared_ptr<NLConstraint> con : sorted_constraints) {
    con->index = sorted_con_index;
    sorted_con_index += 1;
  }
  current_con_ndx = constraints.size();

  std::vector<std::shared_ptr<NLConstraint>> nonlinear_constraints;
  std::vector<std::shared_ptr<NLConstraint>> linear_constraints;
  for (std::shared_ptr<NLConstraint> con : sorted_constraints) {
    if (con->is_nonlinear()) {
      nonlinear_constraints.push_back(con);
    } else {
      linear_constraints.push_back(con);
    }
  }

  // first order the variables and gather some problem statistics

  int jac_nnz = 0;
  int grad_obj_nnz = 0;
  int n_eq_cons = 0;
  int n_range_cons = 0;

  std::vector<double> con_lower;
  std::vector<double> con_upper;
  std::vector<int> n_vars_per_con;
  std::vector<int> con_type;
  std::vector<std::shared_ptr<NLConstraint>> all_cons;
  std::vector<std::shared_ptr<NLConstraint>> active_nonlinear_cons;
  std::vector<std::shared_ptr<NLConstraint>> active_linear_cons;
  std::map<std::string, int> external_function_indices;

  std::shared_ptr<NLObjective> nl_objective =
      std::dynamic_pointer_cast<NLObjective>(objective);

  for (std::shared_ptr<Var> v : *(nl_objective->all_vars)) {
    v->index = -1;
    grad_obj_nnz += 1;
  }

  for (std::shared_ptr<ExternalOperator> n :
       *(nl_objective->external_operators)) {
    std::map<std::string, int>::iterator it =
        external_function_indices.find(n->function_name);
    if (it != external_function_indices.end()) {
      n->external_function_index = it->second;
    } else {
      n->external_function_index = external_function_indices.size();
      external_function_indices[n->function_name] = n->external_function_index;
    }
  }

  // it is important that nonlinear constraints come first here
  bool has_unfixed_vars;
  for (std::shared_ptr<NLConstraint> con : nonlinear_constraints) {
    if (con->active) {
      has_unfixed_vars = false;
      for (std::shared_ptr<Var> v : *(con->all_vars)) {
        if (!(v->fixed)) {
          has_unfixed_vars = true;
          break;
        }
      }
      if (has_unfixed_vars) {
        active_nonlinear_cons.push_back(con);
        all_cons.push_back(con);
      }
      for (std::shared_ptr<ExternalOperator> n : *(con->external_operators)) {
        std::map<std::string, int>::iterator it =
            external_function_indices.find(n->function_name);
        if (it != external_function_indices.end()) {
          n->external_function_index = it->second;
        } else {
          n->external_function_index = external_function_indices.size();
          external_function_indices[n->function_name] =
              n->external_function_index;
        }
      }
    }
  }
  for (std::shared_ptr<NLConstraint> con : linear_constraints) {
    if (con->active) {
      has_unfixed_vars = false;
      for (std::shared_ptr<Var> v : *(con->all_vars)) {
        if (!(v->fixed)) {
          has_unfixed_vars = true;
          break;
        }
      }
      if (has_unfixed_vars) {
        active_linear_cons.push_back(con);
        all_cons.push_back(con);
      }
    }
  }

  unsigned int con_ndx = 0;
  double con_lb;
  double con_ub;
  double body_constant_val;
  int _n_con_vars;
  unsigned int _v_ndx;
  for (std::shared_ptr<NLConstraint> con : all_cons) {
    con_lb = con->lb->evaluate();
    con_ub = con->ub->evaluate();
    body_constant_val = con->constant_expr->evaluate();
    if (con_lb == con_ub) {
      n_eq_cons += 1;
      con_type.push_back(4);
      con_lb -= body_constant_val;
      con_ub = con_lb;
    } else {
      n_range_cons += 1;
      if (con_lb > -inf && con_ub < inf) {
        con_type.push_back(0);
        con_lb -= body_constant_val;
        con_ub -= body_constant_val;
      } else if (con_lb > -inf) {
        con_type.push_back(2);
        con_lb -= body_constant_val;
      } else if (con_ub < inf) {
        con_type.push_back(1);
        con_ub -= body_constant_val;
      } else {
        con_type.push_back(3);
      }
    }
    _n_con_vars = 0;
    _v_ndx = 0;
    for (std::shared_ptr<Var> v : *(con->all_vars)) {
      v->index = -1;
      jac_nnz += 1;
      _n_con_vars += 1;
    }
    n_vars_per_con.push_back(_n_con_vars);
    con_lower.push_back(con_lb);
    con_upper.push_back(con_ub);
    con_ndx += 1;
  }

  // -1 means not visited yet
  // -2 means nonlinear in objective only
  // -3 means nonlinear in constraints only
  // -4 means nonlinear in both
  // -5 means linear
  std::vector<std::shared_ptr<Var>> nl_vars_in_both;
  std::vector<std::shared_ptr<Var>> nl_vars_in_cons;
  std::vector<std::shared_ptr<Var>> nl_vars_in_obj_or_cons;
  std::vector<std::shared_ptr<Var>> nl_vars_just_in_cons;
  std::vector<std::shared_ptr<Var>> nl_vars_just_in_obj;
  std::vector<std::shared_ptr<Var>> linear_vars;
  for (std::shared_ptr<Var> v : *(nl_objective->nonlinear_vars)) {
    v->index = -2;
    nl_vars_in_obj_or_cons.push_back(v);
  }

  for (std::shared_ptr<NLConstraint> con : active_nonlinear_cons) {
    for (std::shared_ptr<Var> v : *(con->nonlinear_vars)) {
      if (v->index == -1) {
        v->index = -3;
        nl_vars_in_obj_or_cons.push_back(v);
        nl_vars_just_in_cons.push_back(v);
        nl_vars_in_cons.push_back(v);
      } else if (v->index == -2) {
        v->index = -4;
        nl_vars_in_cons.push_back(v);
        nl_vars_in_both.push_back(v);
      }
    }
  }

  for (std::shared_ptr<NLConstraint> con : active_linear_cons) {
    for (std::shared_ptr<Var> v : *(con->linear_vars)) {
      if (v->index == -1) {
        v->index = -5;
        linear_vars.push_back(v);
      }
    }
  }

  for (std::shared_ptr<NLConstraint> con : active_nonlinear_cons) {
    for (std::shared_ptr<Var> v : *(con->linear_vars)) {
      if (v->index == -1) {
        v->index = -5;
        linear_vars.push_back(v);
      }
    }
  }

  for (std::shared_ptr<Var> v : *(nl_objective->linear_vars)) {
    if (v->index == -1) {
      v->index = -5;
      linear_vars.push_back(v);
    }
  }

  for (std::shared_ptr<Var> v : *(nl_objective->nonlinear_vars)) {
    if (v->index == -2) {
      nl_vars_just_in_obj.push_back(v);
    }
  }

  int ndx = 0;
  std::vector<std::shared_ptr<Var>> all_vars;
  for (std::shared_ptr<Var> v : nl_vars_in_both) {
    v->index = ndx;
    all_vars.push_back(v);
    ++ndx;
  }

  if (nl_objective->nonlinear_vars->size() > nl_vars_in_cons.size()) {
    for (std::shared_ptr<Var> v : nl_vars_just_in_cons) {
      v->index = ndx;
      all_vars.push_back(v);
      ++ndx;
    }
    for (std::shared_ptr<Var> v : nl_vars_just_in_obj) {
      v->index = ndx;
      all_vars.push_back(v);
      ++ndx;
    }
  } else {
    for (std::shared_ptr<Var> v : nl_vars_just_in_obj) {
      v->index = ndx;
      all_vars.push_back(v);
      ++ndx;
    }
    for (std::shared_ptr<Var> v : nl_vars_just_in_cons) {
      v->index = ndx;
      all_vars.push_back(v);
      ++ndx;
    }
  }

  for (std::shared_ptr<Var> v : linear_vars) {
    v->index = ndx;
    all_vars.push_back(v);
    ++ndx;
  }

  // now write the header

  int n_vars = all_vars.size();
  if (n_vars == 0) {
    throw py::value_error("there are not any unfixed variables in the problem");
  }

  f << "g3 1 1 0\n";
  f << n_vars << " ";
  f << all_cons.size() << " ";
  f << "1 " << n_range_cons << " " << n_eq_cons << " 0\n";
  f << active_nonlinear_cons.size() << " ";
  if (nl_objective->is_nonlinear()) {
    f << "1\n";
  } else {
    f << "0\n";
  }
  f << "0 0\n";
  if (nl_vars_just_in_obj.size() == 0) {
    f << nl_vars_in_cons.size() << " " << nl_vars_in_both.size() << " "
      << nl_vars_in_both.size() << "\n";
  } else {
    f << nl_vars_in_cons.size() << " " << nl_vars_in_obj_or_cons.size() << " "
      << nl_vars_in_both.size() << "\n";
  }
  f << "0 " << external_function_indices.size() << " 0 1\n";
  f << "0 0 0 0 0\n";
  f << jac_nnz << " " << grad_obj_nnz << "\n";
  f << "0 0\n";
  f << "0 0 0 0 0\n";

  // now write the names of the external functions
  for (std::map<std::string, int>::iterator it =
           external_function_indices.begin();
       it != external_function_indices.end(); ++it) {
    f << "F" << it->second << " 1 -1 " << it->first << "\n";
  }

  // now write the nonlinear parts of the constraints in prefix notation
  ndx = 0;
  for (std::shared_ptr<NLConstraint> con : active_nonlinear_cons) {
    f << "C" << ndx << "\n";
    for (std::shared_ptr<Node> &node : *(con->nonlinear_prefix_notation)) {
      node->write_nl_string(f);
    }
    ++ndx;
  }

  for (std::shared_ptr<NLConstraint> con : active_linear_cons) {
    f << "C" << ndx << "\n"
      << "n0\n";
    ++ndx;
  }

  // now write the nonlinear part of the objective in prefix notation
  f << "O0"
    << " " << nl_objective->sense << "\n";
  if (nl_objective->is_nonlinear()) {
    f << "o0\n";
    for (std::shared_ptr<Node> &node :
         *(nl_objective->nonlinear_prefix_notation)) {
      node->write_nl_string(f);
    }
    f << "n";
    f << nl_objective->constant_expr->evaluate() << "\n";
  } else {
    f << "n";
    f << nl_objective->constant_expr->evaluate() << "\n";
  }

  // now write initial variable values
  f << "x " << n_vars << "\n";
  for (std::shared_ptr<Var> v : all_vars) {
    f << v->index << " " << v->value << "\n";
  }

  // now write the constraint bounds
  f << "r\n";
  int _con_type;
  for (con_ndx = 0; con_ndx < all_cons.size(); ++con_ndx) {
    _con_type = con_type[con_ndx];
    if (_con_type == 0) {
      f << "0 " << con_lower[con_ndx] << " " << con_upper[con_ndx] << "\n";
    } else if (_con_type == 1) {
      f << "1 " << con_upper[con_ndx] << "\n";
    } else if (_con_type == 2) {
      f << "2 " << con_lower[con_ndx] << "\n";
    } else if (_con_type == 3) {
      f << "3\n";
    } else {
      f << "4 " << con_lower[con_ndx] << "\n";
    }
  }

  // now write variable bounds
  f << "b\n";
  double v_lb;
  double v_ub;
  Domain v_domain;
  for (std::shared_ptr<Var> v : all_vars) {
    v_lb = v->get_lb();
    v_ub = v->get_ub();
    v_domain = v->get_domain();
    if (v->fixed) {
      f << "4 " << v->value << "\n";
    } else if (v_domain != continuous) {
      throw py::value_error(
          "NLWriter currently only supports continuous variables.");
    } else if (v_lb == v_ub) {
      f << "4 " << v_lb << "\n";
    } else if (v_lb > -inf && v_ub < inf) {
      f << "0 " << v_lb << " " << v_ub << "\n";
    } else if (v_lb > -inf) {
      f << "2 " << v_lb << "\n";
    } else if (v_ub < inf) {
      f << "1 " << v_ub << "\n";
    } else {
      f << "3\n";
    }
  }

  // now write the jacobian column counts
  std::vector<int> referenced_variables;
  for (std::shared_ptr<Var> v : all_vars) {
    referenced_variables.push_back(0);
  }
  for (std::shared_ptr<NLConstraint> con : all_cons) {
    for (std::shared_ptr<Var> v : *(con->all_vars)) {
      referenced_variables[v->index] += 1;
    }
  }

  int cumulative = 0;
  f << "k" << (all_vars.size() - 1) << "\n";
  for (_v_ndx = 0; _v_ndx < (all_vars.size() - 1); ++_v_ndx) {
    cumulative += referenced_variables[_v_ndx];
    f << cumulative << "\n";
  }

  // now write the linear part of the jacobian
  con_ndx = 0;
  for (std::shared_ptr<NLConstraint> con : all_cons) {
    std::vector<std::pair<std::shared_ptr<Var>, double>> sorted_vars;
    _v_ndx = 0;
    for (std::shared_ptr<Var> v : *(con->all_vars)) {
      sorted_vars.push_back(std::make_pair(
          v, con->all_linear_coefficients->at(_v_ndx)->evaluate()));
      _v_ndx += 1;
    }
    std::sort(sorted_vars.begin(), sorted_vars.end(), variable_sorter);

    f << "J" << con_ndx << " " << n_vars_per_con[con_ndx] << "\n";
    for (std::pair<std::shared_ptr<Var>, double> p : sorted_vars) {
      f << p.first->index << " " << p.second << "\n";
    }
    con_ndx += 1;
  }

  // now write the linear part of the gradient of the objective
  std::vector<std::pair<std::shared_ptr<Var>, double>> sorted_obj_vars;
  _v_ndx = 0;
  for (std::shared_ptr<Var> v : *(nl_objective->all_vars)) {
    sorted_obj_vars.push_back(std::make_pair(
        v, nl_objective->all_linear_coefficients->at(_v_ndx)->evaluate()));
    _v_ndx += 1;
  }
  std::sort(sorted_obj_vars.begin(), sorted_obj_vars.end(), variable_sorter);

  if (sorted_obj_vars.size() > 0) {
    f << "G0 " << grad_obj_nnz << "\n";
    for (std::pair<std::shared_ptr<Var>, double> p : sorted_obj_vars) {
      f << p.first->index << " " << p.second << "\n";
    }
  }

  f.close();

  solve_vars = all_vars;
  solve_cons = all_cons;
}

std::vector<std::shared_ptr<Var>> NLWriter::get_solve_vars() {
  return solve_vars;
}

std::vector<std::shared_ptr<NLConstraint>> NLWriter::get_solve_cons() {
  return solve_cons;
}

void process_nl_constraints(NLWriter *nl_writer, PyomoExprTypes &expr_types,
                            py::list cons, py::dict var_map, py::dict param_map,
                            py::dict active_constraints, py::dict con_map,
                            py::dict rev_con_map) {
  py::object generate_standard_repn =
      py::module_::import("pyomo.repn.standard_repn")
          .attr("generate_standard_repn");
  py::object repn;
  std::shared_ptr<ExpressionBase> _const;
  std::vector<std::shared_ptr<Var>> lin_vars;
  std::vector<std::shared_ptr<ExpressionBase>> lin_coefs;
  std::shared_ptr<ExpressionBase> nonlin_expr;
  std::shared_ptr<NLConstraint> nl_con;
  py::tuple lower_body_upper;
  py::handle c_lb;
  py::handle c_ub;
  py::handle repn_nonlinear_expr;

  for (py::handle c : cons) {
    lower_body_upper = active_constraints[c];
    repn = generate_standard_repn(
        lower_body_upper[1], "compute_values"_a = false, "quadratic"_a = false);
    _const = appsi_expr_from_pyomo_expr(repn.attr("constant"), var_map,
                                        param_map, expr_types);
    lin_coefs.clear();
    lin_vars.clear();
    for (py::handle coef : repn.attr("linear_coefs")) {
      lin_coefs.push_back(
          appsi_expr_from_pyomo_expr(coef, var_map, param_map, expr_types));
    }
    for (py::handle v : repn.attr("linear_vars")) {
      lin_vars.push_back(
          var_map[expr_types.id(v)].cast<std::shared_ptr<Var>>());
    }
    repn_nonlinear_expr = repn.attr("nonlinear_expr");
    if (repn_nonlinear_expr.is(py::none())) {
      nonlin_expr = std::make_shared<Constant>(0);
    } else {
      nonlin_expr = appsi_expr_from_pyomo_expr(repn_nonlinear_expr, var_map,
                                               param_map, expr_types);
    }
    nl_con = std::make_shared<NLConstraint>(_const, lin_coefs, lin_vars,
                                            nonlin_expr);

    c_lb = lower_body_upper[0];
    c_ub = lower_body_upper[2];
    if (!(c_lb.is(py::none()))) {
      nl_con->lb =
          appsi_expr_from_pyomo_expr(c_lb, var_map, param_map, expr_types);
    }
    if (!(c_ub.is(py::none()))) {
      nl_con->ub =
          appsi_expr_from_pyomo_expr(c_ub, var_map, param_map, expr_types);
    }
    nl_writer->add_constraint(nl_con);
    con_map[c] = py::cast(nl_con);
    rev_con_map[py::cast(nl_con)] = c;
  }
}
