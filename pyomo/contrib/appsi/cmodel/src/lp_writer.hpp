#include "nl_writer.hpp"


class LPBase;
class LPConstraint;
class LPObjective;
class LPWriter;


extern double inf;


class LPBase
{
public:
  LPBase(std::shared_ptr<ExpressionBase> _constant_expr,
	 std::vector<std::shared_ptr<ExpressionBase> > &_linear_coefficients,
	 std::vector<std::shared_ptr<Var> > &_linear_vars,
	 std::vector<std::shared_ptr<ExpressionBase> > &_quadratic_coefficients,
	 std::vector<std::shared_ptr<Var> > &_quadratic_vars_1,
	 std::vector<std::shared_ptr<Var> > &_quadratic_vars_2);
  virtual ~LPBase() = default;
  std::shared_ptr<ExpressionBase> constant_expr;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase> > > linear_coefficients;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > linear_vars;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase> > > quadratic_coefficients;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > quadratic_vars_1;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > quadratic_vars_2;
  std::string name;
};


class LPObjective: public LPBase
{
public:
  LPObjective(std::shared_ptr<ExpressionBase> _constant_expr,
	      std::vector<std::shared_ptr<ExpressionBase> > _linear_coefficients,
	      std::vector<std::shared_ptr<Var> > _linear_vars,
	      std::vector<std::shared_ptr<ExpressionBase> > _quadratic_coefficients,
	      std::vector<std::shared_ptr<Var> > _quadratic_vars_1,
	      std::vector<std::shared_ptr<Var> > _quadratic_vars_2) : LPBase(_constant_expr,
									     _linear_coefficients,
									     _linear_vars,
									     _quadratic_coefficients,
									     _quadratic_vars_1,
									     _quadratic_vars_2) {}
  int sense = 0; // 0 means min; 1 means max
};


class LPConstraint: public LPBase
{
public:
  LPConstraint(std::shared_ptr<ExpressionBase> _constant_expr,
	       std::vector<std::shared_ptr<ExpressionBase> > _linear_coefficients,
	       std::vector<std::shared_ptr<Var> > _linear_vars,
	       std::vector<std::shared_ptr<ExpressionBase> > _quadratic_coefficients,
	       std::vector<std::shared_ptr<Var> > _quadratic_vars_1,
	       std::vector<std::shared_ptr<Var> > _quadratic_vars_2) : LPBase(_constant_expr,
									      _linear_coefficients,
									      _linear_vars,
									      _quadratic_coefficients,
									      _quadratic_vars_1,
									      _quadratic_vars_2) {}
  std::shared_ptr<ExpressionBase> lb = std::make_shared<Constant>(-inf);
  std::shared_ptr<ExpressionBase> ub = std::make_shared<Constant>(inf);
  bool active = true;
  int index = -1;
};


class LPWriter
{
public:
  LPWriter() = default;
  std::shared_ptr<LPObjective> objective;
  std::shared_ptr<std::set<std::shared_ptr<LPConstraint> > > constraints = std::make_shared<std::set<std::shared_ptr<LPConstraint> > >();
  std::vector<std::shared_ptr<LPConstraint> > solve_cons;
  std::vector<std::shared_ptr<Var> > solve_vars;
  void write(std::string filename);
  void add_constraint(std::shared_ptr<LPConstraint>);
  void remove_constraint(std::shared_ptr<LPConstraint>);
  std::vector<std::shared_ptr<LPConstraint> > get_solve_cons();
  std::vector<std::shared_ptr<Var> > get_solve_vars();
  int current_cons_index = 0;
};

void process_lp_constraints(py::list, py::object);
