#include "model.hpp"


bool constraint_sorter(std::shared_ptr<Constraint> c1, std::shared_ptr<Constraint> c2)
{
  return c1->index < c2->index;
}


bool var_sorter(std::shared_ptr<Var> v1, std::shared_ptr<Var> v2)
{
  return v1->index < v2->index;
}


Objective::Objective(std::shared_ptr<ExpressionBase> _expr)
{
  expr = _expr;
}


Constraint::Constraint(std::shared_ptr<ExpressionBase> _lb, std::shared_ptr<ExpressionBase> _body, std::shared_ptr<ExpressionBase> _ub)
{
  lb = _lb;
  body = _body;
  ub = _ub;
  variables = body->identify_variables();
}


void Constraint::perform_fbbt(double feasibility_tol, double integer_tol, double improvement_tol,
		    std::set<std::shared_ptr<Var> >& improved_vars, bool immediate_update, bool multiple_threads,
		    double* var_lbs, double* var_ubs, std::mutex* mtx)
{
  double body_lb;
  double body_ub;
  double* lbs = new double[1];
  double* ubs = new double[1];

  if (body->is_expression_type())
    {
      std::shared_ptr<Expression> e = std::dynamic_pointer_cast<Expression>(body);
      delete[] lbs;
      delete[] ubs;
      lbs = new double[e->n_operators];
      ubs = new double[e->n_operators];
      e->propagate_bounds_forward(lbs, ubs, feasibility_tol, integer_tol);
    }

  body_lb = body->get_lb_from_array(lbs);
  body_ub = body->get_ub_from_array(ubs);

  double con_lb = lb->evaluate();
  double con_ub = ub->evaluate();

  if (body_lb > con_ub + feasibility_tol || body_ub < con_ub - feasibility_tol)
    {
      throw py::value_error("Infeasible constraint");
    }

  if (con_lb > body_lb || con_ub < body_ub) // otherwise the constraint is always satisfied
    {
      if (con_lb > body_lb)
	{
	  body_lb = con_lb;
	}
      if (con_ub < body_ub)
	{
	  body_ub = con_ub;
	}
      body->set_bounds_in_array(body_lb, body_ub, lbs, ubs, feasibility_tol, integer_tol, improvement_tol, improved_vars,
				immediate_update, multiple_threads, var_lbs, var_ubs, mtx);
      if (body->is_expression_type())
	{
	  std::shared_ptr<Expression> e = std::dynamic_pointer_cast<Expression>(body);
	  e->propagate_bounds_backward(lbs, ubs, feasibility_tol, integer_tol, improvement_tol, improved_vars,
				       immediate_update, multiple_threads, var_lbs, var_ubs, mtx);
	}
    }
  delete[] lbs;
  delete[] ubs;
}


Model::Model()
{
  constraints = std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*>(constraint_sorter);
  var_to_con_map = std::map<std::shared_ptr<Var>,
			    std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*>,
			    decltype(var_sorter)*>(var_sorter);
}


void Model::add_constraint(std::shared_ptr<Constraint> con)
{
  con->index = current_con_ndx;
  current_con_ndx += 1;
  constraints.insert(con);
  for (std::shared_ptr<Var>& v : *(con->variables))
    {
      if (var_to_con_map.count(v) == 0)
	{
	  v->index = current_var_ndx;
	  current_var_ndx += 1;
	  var_to_con_map.insert(std::pair<std::shared_ptr<Var>, std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*> >(v, std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*>(constraint_sorter)));
	}
      var_to_con_map[v].insert(con);
    }
  needs_update = true;
}


void Model::remove_constraint(std::shared_ptr<Constraint> con)
{
  for (std::shared_ptr<Var>& v : *(con->variables))
    {
      var_to_con_map[v].erase(con);
      if (var_to_con_map[v].size() == 0)
	{
	  var_to_con_map.erase(v);
	  v->index = -1;
	}
    }
  constraints.erase(con);
  con->index = -1;
  needs_update = true;
}


void Model::update()
{
  current_con_ndx = 0;
  for (const std::shared_ptr<Constraint>& c : constraints)
    {
      c->index = current_con_ndx;
      current_con_ndx += 1;
    }

  current_var_ndx = 0;
  for (auto const& p : var_to_con_map)
    {
      p.first->index = current_var_ndx;
      current_var_ndx += 1;
    }
  
  needs_update = false;
}


void _fbbt_thread(std::vector<std::shared_ptr<Constraint> >* con_vector, int thread_ndx, int num_threads, double feasibility_tol,
		  double integer_tol, double improvement_tol, std::set<std::shared_ptr<Var> >* improved_vars,
		  bool immediate_update, bool multiple_threads, double* var_lbs, double* var_ubs, std::mutex* mtx)
{
  unsigned int con_ndx = thread_ndx;
  while (con_ndx < con_vector->size())
    {
      (*con_vector)[con_ndx]->perform_fbbt(feasibility_tol, integer_tol, improvement_tol, *improved_vars,
					   immediate_update, multiple_threads, var_lbs, var_ubs, mtx);
      con_ndx += num_threads;
    }
}


void _fbbt_iter(std::vector<std::shared_ptr<Constraint> >& cons, double feasibility_tol, double integer_tol,
		int num_threads, double improvement_tol, std::set<std::shared_ptr<Var> >& improved_vars,
		bool immediate_update, double* var_lbs, double* var_ubs, std::mutex* mtx)
{
  bool multiple_threads;
  if (num_threads > 1)
    multiple_threads = true;
  else
    multiple_threads = false;

  std::vector<std::thread> thread_vec(num_threads);
  for (int thread_ndx=0; thread_ndx < num_threads; ++thread_ndx)
    {
      thread_vec[thread_ndx] = std::thread(_fbbt_thread, &cons, thread_ndx, num_threads, feasibility_tol, integer_tol,
					   improvement_tol, &improved_vars, immediate_update, multiple_threads, var_lbs,
					   var_ubs, mtx);
    }

  for (int thread_ndx=0; thread_ndx < num_threads; ++thread_ndx)
    {
      thread_vec[thread_ndx].join();
    }
}


unsigned int Model::perform_fbbt_on_cons(std::vector<std::shared_ptr<Constraint> >& seed_cons, double feasibility_tol, double integer_tol,
					 double improvement_tol, int max_iter, int num_threads, bool immediate_update)
{
  if (needs_update)
    update();

  std::set<std::shared_ptr<Var> > improved_vars_set;
  std::vector<std::shared_ptr<Var> > improved_vars_vec;
  int n_vars = var_to_con_map.size();
  double* var_lbs = new double[n_vars];
  double* var_ubs = new double[n_vars];
  std::mutex* mtx = new std::mutex[n_vars];

  if (!immediate_update)
    {
      for (auto const& p : var_to_con_map)
	{
	  assert (p.first->lb->is_leaf());
	  var_lbs[p.first->index] = p.first->get_lb();
	  var_ubs[p.first->index] = p.first->get_ub();
	}
    }

  std::vector<std::shared_ptr<Constraint> > cons_to_fbbt = seed_cons;
  std::set<std::shared_ptr<Constraint> > cons_to_fbbt_set;
  unsigned int _iter = 0;
  while (_iter < max_iter*constraints.size() && cons_to_fbbt.size() > 0)
    {
      _iter += cons_to_fbbt.size();
      _fbbt_iter(cons_to_fbbt, feasibility_tol, integer_tol, num_threads, improvement_tol, improved_vars_set,
		 immediate_update, var_lbs, var_ubs, mtx);

      if (!immediate_update)
	{
	  for (auto const& p : var_to_con_map)
	    {
	      assert (p.first->lb->is_leaf());
	      std::dynamic_pointer_cast<Leaf>(p.first->lb)->value = var_lbs[p.first->index];
	      std::dynamic_pointer_cast<Leaf>(p.first->ub)->value = var_ubs[p.first->index];
	    }
	}
      
      cons_to_fbbt.clear();
      cons_to_fbbt_set.clear();

      improved_vars_vec.clear();
      for (const std::shared_ptr<Var>& v : improved_vars_set)
	improved_vars_vec.push_back(v);
      std::sort(improved_vars_vec.begin(), improved_vars_vec.end(), var_sorter);
      improved_vars_set.clear();

      for (const std::shared_ptr<Var>& v : improved_vars_vec)
	{
	  for (const std::shared_ptr<Constraint>& c : var_to_con_map[v])
	    {
	      if (cons_to_fbbt_set.count(c) == 0)
		{
		  cons_to_fbbt_set.insert(c);
		  cons_to_fbbt.push_back(c);
		}
	    }
	}
    }

  delete[] var_lbs;
  delete[] var_ubs;
  delete[] mtx;

  return _iter;
}


unsigned int Model::perform_fbbt_with_seed(std::shared_ptr<Var> seed_var, double feasibility_tol, double integer_tol, double improvement_tol,
					   int max_iter, int num_threads, bool immediate_update)
{
  if (needs_update)
    update();

  std::set<std::shared_ptr<Constraint>, decltype(constraint_sorter)*>& seed_con_set = var_to_con_map[seed_var];
  int n_cons = seed_con_set.size();
  std::vector<std::shared_ptr<Constraint> > con_vector(n_cons);

  unsigned int ndx = 0;
  for (const std::shared_ptr<Constraint>& c : seed_con_set)
    {
      con_vector[ndx] = c;
      ndx += 1;
    }

  return perform_fbbt_on_cons(con_vector, feasibility_tol, integer_tol, improvement_tol, max_iter, num_threads, immediate_update);
}


unsigned int Model::perform_fbbt(double feasibility_tol, double integer_tol, double improvement_tol,
				 int max_iter, int num_threads, bool immediate_update)
{
  if (needs_update)
    update();

  int n_cons = constraints.size();
  std::vector<std::shared_ptr<Constraint> > con_vector(n_cons);

  unsigned int ndx = 0;
  for (const std::shared_ptr<Constraint>& c : constraints)
    {
      con_vector[ndx] = c;
      ndx += 1;
    }

  return perform_fbbt_on_cons(con_vector, feasibility_tol, integer_tol, improvement_tol, max_iter, num_threads, immediate_update);
}


void process_constraints(Model* model,
			 PyomoExprTypes& expr_types,
			 py::list cons,
			 py::dict var_map,
			 py::dict param_map,
			 py::dict active_constraints,
			 py::dict con_map,
			 py::dict rev_con_map)
{
  std::shared_ptr<Constraint> ccon;
  std::shared_ptr<ExpressionBase> ccon_lb;
  std::shared_ptr<ExpressionBase> ccon_ub;
  std::shared_ptr<ExpressionBase> ccon_body;
  py::tuple lower_body_upper;
  py::handle con_lb;
  py::handle con_ub;
  py::handle con_body;

  for (py::handle c : cons)
    {
      lower_body_upper = active_constraints[c];
      con_lb = lower_body_upper[0];
      con_body = lower_body_upper[1];
      con_ub = lower_body_upper[2];

      ccon_body = appsi_expr_from_pyomo_expr(con_body, var_map, param_map, expr_types);

      if (con_lb.is(py::none()))
	{
	  ccon_lb = std::make_shared<Constant>(0);
	}
      else
	{
	  ccon_lb = appsi_expr_from_pyomo_expr(con_lb, var_map, param_map, expr_types);
	}

      if (con_ub.is(py::none()))
	{
	  ccon_ub = std::make_shared<Constant>(0);
	}
      else
	{
	  ccon_ub = appsi_expr_from_pyomo_expr(con_ub, var_map, param_map, expr_types);
	}

      ccon = std::make_shared<Constraint>(ccon_lb, ccon_body, ccon_ub);
      model->add_constraint(ccon);
      con_map[c] = py::cast(ccon);
      rev_con_map[py::cast(ccon)] = c;
    }
}
