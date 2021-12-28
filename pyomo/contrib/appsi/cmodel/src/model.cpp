#include "model.hpp"


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


void Constraint::perform_fbbt(double feasibility_tol, double integer_tol)
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
      body->set_bounds_in_array(body_lb, body_ub, lbs, ubs, feasibility_tol, integer_tol);
      if (body->is_expression_type())
	{
	  std::shared_ptr<Expression> e = std::dynamic_pointer_cast<Expression>(body);
	  e->propagate_bounds_backward(lbs, ubs, feasibility_tol, integer_tol);
	}
    }
  delete[] lbs;
  delete[] ubs;
}


void Model::add_constraint(std::shared_ptr<Constraint> con)
{
  constraints.insert(con);
}


void Model::remove_constraint(std::shared_ptr<Constraint> con)
{
  constraints.erase(con);
}


void _fbbt_thread(std::vector<std::shared_ptr<Constraint> >* con_vector, int thread_ndx, int num_threads, double feasibility_tol, double integer_tol)
{
  unsigned int con_ndx = thread_ndx;
  while (con_ndx < con_vector->size())
    {
      (*con_vector)[con_ndx]->perform_fbbt(feasibility_tol, integer_tol);
      con_ndx += num_threads;
    }
}


void _fbbt_iter(std::vector<std::shared_ptr<Constraint> >& cons, double feasibility_tol, double integer_tol, int num_threads)
{
  std::vector<std::thread> thread_vec(num_threads);
  for (int thread_ndx=0; thread_ndx < num_threads; ++thread_ndx)
    {
      thread_vec[thread_ndx] = std::thread(_fbbt_thread, &cons, thread_ndx, num_threads, feasibility_tol, integer_tol);
    }

  for (int thread_ndx=0; thread_ndx < num_threads; ++thread_ndx)
    {
      thread_vec[thread_ndx].join();
    }
}


void perform_fbbt_on_cons(std::vector<std::shared_ptr<Constraint> >& cons, double feasibility_tol, double integer_tol, double improvement_tol, int max_iter, int num_threads)
{
  std::vector<std::shared_ptr<Var> > all_variables;
  std::set<std::shared_ptr<Var> > var_set;
  std::map<std::shared_ptr<Var>, std::vector<std::shared_ptr<Constraint> > > var_to_con_map;

  for (std::shared_ptr<Constraint>& c : cons)
    {
      for (std::shared_ptr<Var>& v : *(c->variables))
	{
	  if (var_set.count(v) == 0)
	    {
	      var_set.insert(v);
	      all_variables.push_back(v);
	    }
	  var_to_con_map[v].push_back(c);
	}
    }

  int n_vars = all_variables.size();
  double* start_lbs = new double[n_vars];
  double* start_ubs = new double[n_vars];

  std::vector<std::shared_ptr<Constraint> > cons_to_fbbt = cons;
  std::set<std::shared_ptr<Var> > improved_vars;
  std::set<std::shared_ptr<Constraint> > cons_to_fbbt_set;
  unsigned int _iter = 0;
  while (_iter < max_iter*cons.size() && cons_to_fbbt.size() > 0)
    {
      _iter += cons_to_fbbt.size();
      for (int v_ndx=0; v_ndx<n_vars; ++v_ndx)
	{
	  start_lbs[v_ndx] = all_variables[v_ndx]->get_lb();
	  start_ubs[v_ndx] = all_variables[v_ndx]->get_ub();
	}
      _fbbt_iter(cons_to_fbbt, feasibility_tol, integer_tol, num_threads);
      cons_to_fbbt.clear();
      improved_vars.clear();
      cons_to_fbbt_set.clear();
      for (int v_ndx=0; v_ndx<n_vars; ++v_ndx)
	{
	  if (all_variables[v_ndx]->get_lb() > start_lbs[v_ndx] + improvement_tol
	      || all_variables[v_ndx]->get_ub() < start_ubs[v_ndx] - improvement_tol)
	    {
	      improved_vars.insert(all_variables[v_ndx]);
	      start_lbs[v_ndx] = all_variables[v_ndx]->get_lb();
	      start_ubs[v_ndx] = all_variables[v_ndx]->get_ub();
	    }
	}
      for (const std::shared_ptr<Var>& v : improved_vars)
	{
	  for (std::shared_ptr<Constraint>& c : var_to_con_map[v])
	    {
	      if (cons_to_fbbt_set.count(c) == 0)
		{
		  cons_to_fbbt_set.insert(c);
		  cons_to_fbbt.push_back(c);
		}
	    }
	}
    }
}


void Model::perform_fbbt(double feasibility_tol, double integer_tol, double improvement_tol, int max_iter, int num_threads)
{
  int n_cons = constraints.size();
  std::vector<std::shared_ptr<Constraint> > con_vector(n_cons);

  unsigned int ndx = 0;
  for (std::shared_ptr<Constraint> c : constraints)
    {
      con_vector[ndx] = c;
      ndx += 1;
    }

  perform_fbbt_on_cons(con_vector, feasibility_tol, integer_tol, improvement_tol, max_iter, num_threads);
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
