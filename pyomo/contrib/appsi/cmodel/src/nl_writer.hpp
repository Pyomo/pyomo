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

#include "model_base.hpp"

class NLBase;
class NLConstraint;
class NLObjective;
class NLWriter;

extern double inf;

class NLBase {
public:
  NLBase(std::shared_ptr<ExpressionBase> _constant_expr,
         std::vector<std::shared_ptr<ExpressionBase>> &_linear_coefficients,
         std::vector<std::shared_ptr<Var>> &_linear_vars,
         std::shared_ptr<ExpressionBase> _nonlinear_expr);
  virtual ~NLBase() = default;
  std::shared_ptr<ExpressionBase> constant_expr;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> nonlinear_vars;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>>
      linear_vars; // these may also be in the nonlinear vars
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> all_vars;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>>
      all_linear_coefficients;
  std::shared_ptr<std::vector<std::shared_ptr<Node>>> nonlinear_prefix_notation;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator>>>
      external_operators;
  bool is_nonlinear();
};

class NLObjective : public NLBase, public Objective {
public:
  NLObjective(std::shared_ptr<ExpressionBase> _constant_expr,
              std::vector<std::shared_ptr<ExpressionBase>> _linear_coefficients,
              std::vector<std::shared_ptr<Var>> _linear_vars,
              std::shared_ptr<ExpressionBase> _nonlinear_expr)
      : NLBase(_constant_expr, _linear_coefficients, _linear_vars,
               _nonlinear_expr) {}
};

class NLConstraint : public NLBase, public Constraint {
public:
  NLConstraint(
      std::shared_ptr<ExpressionBase> _constant_expr,
      std::vector<std::shared_ptr<ExpressionBase>> _linear_coefficients,
      std::vector<std::shared_ptr<Var>> _linear_vars,
      std::shared_ptr<ExpressionBase> _nonlinear_expr)
      : NLBase(_constant_expr, _linear_coefficients, _linear_vars,
               _nonlinear_expr) {}
};

class NLWriter : public Model {
public:
  NLWriter() = default;
  std::vector<std::shared_ptr<Var>> solve_vars;
  std::vector<std::shared_ptr<NLConstraint>> solve_cons;
  void write(std::string filename);
  std::vector<std::shared_ptr<Var>> get_solve_vars();
  std::vector<std::shared_ptr<NLConstraint>> get_solve_cons();
};

void process_nl_constraints(NLWriter *nl_writer, PyomoExprTypes &expr_types,
                            py::list cons, py::dict var_map, py::dict param_map,
                            py::dict active_constraints, py::dict con_map,
                            py::dict rev_con_map);
