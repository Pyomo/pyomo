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
  int index = -1;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > variables;
  void perform_fbbt(double feasibility_tol, double integer_tol, double improvement_tol,
		    std::set<std::shared_ptr<Var> >& improved_vars, bool immediate_update, bool multiple_threads,
		    double* var_lbs, double* var_ubs, std::mutex* mtx);
};


bool constraint_sorter(std::shared_ptr<Constraint> c1, std::shared_ptr<Constraint> c2);
bool var_sorter(std::shared_ptr<Var> v1, std::shared_ptr<Var> v2);


class Model
{
public:
  Model();
  ~Model() = default;
  std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*> constraints;
  std::map<std::shared_ptr<Var>,
	   std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*>,
	   decltype(var_sorter)*> var_to_con_map;
  std::shared_ptr<Objective> objective;
  void add_constraint(std::shared_ptr<Constraint>);
  void remove_constraint(std::shared_ptr<Constraint>);
  unsigned int perform_fbbt_on_cons(std::vector<std::shared_ptr<Constraint> >& seed_cons, double feasibility_tol, double integer_tol,
				    double improvement_tol, int max_iter, int num_threads, bool immediate_update);
  unsigned int perform_fbbt_with_seed(std::shared_ptr<Var> seed_var, double feasibility_tol, double integer_tol, double improvement_tol,
				      int max_iter, int num_threads, bool immediate_update);
  unsigned int perform_fbbt(double feasibility_tol, double integer_tol, double improvement_tol,
			    int max_iter, int num_threads, bool immediate_update);
  int current_con_ndx = 0;
  int current_var_ndx = 0;
  bool needs_update = false;
  void update();
};


void process_constraints(Model* model,
			 PyomoExprTypes& expr_types,
			 py::list cons,
			 py::dict var_map,
			 py::dict param_map,
			 py::dict active_constraints,
			 py::dict con_map,
			 py::dict rev_con_map);
