#include "model_base.hpp"

class LPBase;
class LPConstraint;
class LPObjective;
class LPWriter;

extern double inf;

class LPBase {
public:
  LPBase() = default;
  virtual ~LPBase() = default;
  std::shared_ptr<ExpressionBase> constant_expr;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>>
      linear_coefficients;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> linear_vars;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase>>>
      quadratic_coefficients;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> quadratic_vars_1;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>> quadratic_vars_2;
};

class LPObjective : public LPBase, public Objective {
public:
  LPObjective() = default;
};

class LPConstraint : public LPBase, public Constraint {
public:
  LPConstraint() = default;
};

class LPWriter : public Model {
public:
  LPWriter() = default;
  std::vector<std::shared_ptr<LPConstraint>> solve_cons;
  std::vector<std::shared_ptr<Var>> solve_vars;
  void write(std::string filename);
  std::vector<std::shared_ptr<LPConstraint>> get_solve_cons();
  std::vector<std::shared_ptr<Var>> get_solve_vars();
};

void process_lp_constraints(py::list, py::object);
std::shared_ptr<LPObjective> process_lp_objective(PyomoExprTypes &expr_types,
                                                  py::object pyomo_obj,
                                                  py::dict var_map,
                                                  py::dict param_map);
