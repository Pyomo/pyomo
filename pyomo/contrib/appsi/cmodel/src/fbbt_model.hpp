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

class FBBTConstraint;
class FBBTObjective;
class FBBTModel;

extern double inf;

class FBBTObjective : public Objective {
public:
  FBBTObjective(std::shared_ptr<ExpressionBase> _expr);
  ~FBBTObjective() = default;
  std::shared_ptr<ExpressionBase> expr;
};

class FBBTConstraint : public Constraint {
public:
  FBBTConstraint(std::shared_ptr<ExpressionBase> _lb,
                 std::shared_ptr<ExpressionBase> _body,
                 std::shared_ptr<ExpressionBase> _ub);
  ~FBBTConstraint();
  std::shared_ptr<ExpressionBase> body;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> variables;
  double *lbs;
  double *ubs;
  void perform_fbbt(double feasibility_tol, double integer_tol,
                    double improvement_tol,
                    std::set<std::shared_ptr<Var>> &improved_vars,
                    bool deactivate_satisfied_constraints);
};

class FBBTModel : public Model {
public:
  FBBTModel() = default;
  ~FBBTModel() = default;
  unsigned int perform_fbbt_on_cons(
      std::vector<std::shared_ptr<FBBTConstraint>> &seed_cons,
      double feasibility_tol, double integer_tol, double improvement_tol,
      int max_iter, bool deactivate_satisfied_constraints,
      std::shared_ptr<std::map<std::shared_ptr<Var>,
                               std::vector<std::shared_ptr<FBBTConstraint>>>>
          var_to_con_map);
  unsigned int perform_fbbt_with_seed(std::shared_ptr<Var> seed_var,
                                      double feasibility_tol,
                                      double integer_tol,
                                      double improvement_tol, int max_iter,
                                      bool deactivate_satisfied_constraints);
  unsigned int perform_fbbt(double feasibility_tol, double integer_tol,
                            double improvement_tol, int max_iter,
                            bool deactivate_satisfied_constraints);
  std::shared_ptr<std::map<std::shared_ptr<Var>,
                           std::vector<std::shared_ptr<FBBTConstraint>>>>
  get_var_to_con_map();
};

void process_fbbt_constraints(FBBTModel *model, PyomoExprTypes &expr_types,
                              py::list cons, py::dict var_map,
                              py::dict param_map, py::dict active_constraints,
                              py::dict con_map, py::dict rev_con_map);
