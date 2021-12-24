#include "lp_writer.hpp"


class Constraint;
class Objective;
class Model;


extern double inf;


class Objective
{
public:
  Objective(std::shared_ptr<ExpressionBase> _expr);
  ~Objective() = default;
  std::shared_ptr<ExpressionBase> expr;
  int sense = 0; // 0 means min; 1 means max
};


class Constraint
{
public:
  Constraint(std::shared_ptr<ExpressionBase> _lb, std::shared_ptr<ExpressionBase> _body, std::shared_ptr<ExpressionBase> _ub);
  ~Constraint() = default;
  std::shared_ptr<ExpressionBase> body;
  std::shared_ptr<ExpressionBase> lb;
  std::shared_ptr<ExpressionBase> ub;
  bool active = true;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > variables;
  void perform_fbbt(double feasibility_tol, double integer_tol);
};


class Model
{
public:
  Model() = default;
  ~Model() = default;
  std::set<std::shared_ptr<Constraint> > constraints;
  std::shared_ptr<Objective> objective;
  void add_constraint(std::shared_ptr<Constraint>);
  void remove_constraint(std::shared_ptr<Constraint>);
  void perform_fbbt(double feasibility_tol, double integer_tol, double improvement_tol, int max_iter, int num_threads);
};


void process_constraints(Model* model,
			 PyomoExprTypes& expr_types,
			 py::list cons,
			 py::dict var_map,
			 py::dict param_map,
			 py::dict active_constraints,
			 py::dict con_map,
			 py::dict rev_con_map);
