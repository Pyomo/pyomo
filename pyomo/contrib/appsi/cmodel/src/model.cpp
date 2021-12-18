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
