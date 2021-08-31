#include "expression.hpp"


class NLBase;
class NLConstraint;
class NLObjective;
class NLWriter;


extern double inf;


class NLBase
{
public:
  NLBase(std::shared_ptr<ExpressionBase> _constant_expr,
	 std::vector<std::shared_ptr<ExpressionBase> > &_linear_coefficients,
	 std::vector<std::shared_ptr<Var> > &_linear_vars,
	 std::shared_ptr<ExpressionBase> _nonlinear_expr);
  virtual ~NLBase() = default;
  std::shared_ptr<ExpressionBase> constant_expr;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > nonlinear_vars;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > linear_vars;  // these may also be in the nonlinear vars
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > all_vars;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase> > > all_linear_coefficients;
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > nonlinear_prefix_notation;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > external_operators;
  bool is_nonlinear();
};


class NLObjective: public NLBase
{
public:
  NLObjective(std::shared_ptr<ExpressionBase> _constant_expr,
	      std::vector<std::shared_ptr<ExpressionBase> > _linear_coefficients,
	      std::vector<std::shared_ptr<Var> > _linear_vars,
	      std::shared_ptr<ExpressionBase> _nonlinear_expr) : NLBase(_constant_expr,
									_linear_coefficients,
									_linear_vars,
									_nonlinear_expr) {}
  int sense = 0; // 0 means min; 1 means max
};


class NLConstraint: public NLBase
{
public:
  NLConstraint(std::shared_ptr<ExpressionBase> _constant_expr,
	       std::vector<std::shared_ptr<ExpressionBase> > _linear_coefficients,
	       std::vector<std::shared_ptr<Var> > _linear_vars,
	       std::shared_ptr<ExpressionBase> _nonlinear_expr) : NLBase(_constant_expr,
									 _linear_coefficients,
									 _linear_vars,
									 _nonlinear_expr) {}
  std::shared_ptr<ExpressionBase> lb = std::make_shared<Constant>(-inf);
  std::shared_ptr<ExpressionBase> ub = std::make_shared<Constant>(inf);
  bool active = true;
  int index = -1;
};


class NLWriter
{
public:
  NLWriter() = default;
  std::shared_ptr<NLObjective> objective;
  std::shared_ptr<std::set<std::shared_ptr<NLConstraint> > > constraints = std::make_shared<std::set<std::shared_ptr<NLConstraint> > >();
  std::vector<std::shared_ptr<Var> > solve_vars;
  std::vector<std::shared_ptr<NLConstraint> > solve_cons;
  void write(std::string filename);
  void add_constraint(std::shared_ptr<NLConstraint>);
  void remove_constraint(std::shared_ptr<NLConstraint>);
  std::vector<std::shared_ptr<Var> > get_solve_vars();
  std::vector<std::shared_ptr<NLConstraint> > get_solve_cons();
  int current_cons_index = 0;
};
