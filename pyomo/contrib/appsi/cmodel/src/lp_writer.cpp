#include "lp_writer.hpp"

LPBase::LPBase(std::shared_ptr<ExpressionBase> _constant_expr,
	       std::vector<std::shared_ptr<ExpressionBase> > &_linear_coefficients,
	       std::vector<std::shared_ptr<Var> > &_linear_vars,
	       std::vector<std::shared_ptr<ExpressionBase> > &_quadratic_coefficients,
	       std::vector<std::shared_ptr<Var> > &_quadratic_vars_1,
	       std::vector<std::shared_ptr<Var> > &_quadratic_vars_2)
{
  constant_expr = _constant_expr;

  linear_coefficients = std::make_shared<std::vector<std::shared_ptr<ExpressionBase> > >();
  linear_vars = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  for (unsigned int ndx=0; ndx<_linear_vars.size(); ++ndx)
    {
      linear_coefficients->push_back(_linear_coefficients[ndx]);
      linear_vars->push_back(_linear_vars[ndx]);
    }

  quadratic_coefficients = std::make_shared<std::vector<std::shared_ptr<ExpressionBase> > >();
  quadratic_vars_1 = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  quadratic_vars_2 = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  for (unsigned int ndx=0; ndx<_quadratic_coefficients.size(); ++ndx)
    {
      quadratic_coefficients->push_back(_quadratic_coefficients[ndx]);
      quadratic_vars_1->push_back(_quadratic_vars_1[ndx]);
      quadratic_vars_2->push_back(_quadratic_vars_2[ndx]);
    }
}


void LPWriter::add_constraint(std::shared_ptr<LPConstraint> con)
{
  con->index = current_cons_index;
  ++current_cons_index;
  constraints->insert(con);
}


void LPWriter::remove_constraint(std::shared_ptr<LPConstraint> con)
{
  con->index = -1;
  constraints->erase(con);
}


bool constraint_sorter(std::shared_ptr<LPConstraint> con1, std::shared_ptr<LPConstraint> con2)
{
  return con1->index < con2->index;
}


void write_expr(std::ofstream &f, std::shared_ptr<LPBase> obj, bool is_objective)
{
  double coef;
  for (unsigned int ndx=0; ndx<obj->linear_coefficients->size(); ++ndx)
    {
      coef = obj->linear_coefficients->at(ndx)->evaluate();
      if (coef >= 0)
	{
	  f << "+";
	}
      else
	{
	  f << "-";
	}
      f << std::abs(coef) << " ";
      f << obj->linear_vars->at(ndx)->name << " \n";
    }
  if (is_objective)
    {
      f << "+1 obj_const \n";
    }

  if (obj->quadratic_coefficients->size() != 0)
    {
      f << "+ [ \n";
      for (unsigned int ndx=0; ndx<obj->quadratic_coefficients->size(); ++ndx)
	{
	  coef = obj->quadratic_coefficients->at(ndx)->evaluate();
	  if (is_objective)
	    {
	      coef *= 2;
	    }
	  if (coef >= 0)
	    {
	      f << "+";
	    }
	  else
	    {
	      f << "-";
	    }
	  f << std::abs(coef) << " ";
	  f << obj->quadratic_vars_1->at(ndx)->name << " * ";
	  f << obj->quadratic_vars_2->at(ndx)->name << " \n";
	}
      f << "] ";
      if (is_objective)
	{
	  f << "/ 2 ";
	}
      f << "\n";
    }
}


void LPWriter::write(std::string filename)
{
  std::ofstream f;
  f.open(filename);
  f.precision(17);

  if (objective->sense == 0)
    {
      f << "minimize\n";
    }
  else
    {
      f << "maximize\n";
    }

  f << objective->name << ": \n";
  write_expr(f, objective, true);

  f << "\ns.t.\n\n";
  
  std::vector<std::shared_ptr<LPConstraint> > sorted_constraints;
  for (std::shared_ptr<LPConstraint> con : *constraints)
    {
      sorted_constraints.push_back(con);
    }
  std::sort(sorted_constraints.begin(), sorted_constraints.end(), constraint_sorter);
  int sorted_con_index = 0;
  for (std::shared_ptr<LPConstraint> con : sorted_constraints)
    {
      con->index = sorted_con_index;
      sorted_con_index += 1;
    }
  current_cons_index = constraints->size();

  std::vector<std::shared_ptr<LPConstraint> > active_constraints;
  for (std::shared_ptr<LPConstraint> con : sorted_constraints)
    {
      if (con->active)
	{
	  active_constraints.push_back(con);
	}
    }

  double con_lb;
  double con_ub;
  double body_constant_val;
  for (std::shared_ptr<LPConstraint> con : active_constraints)
    {
      con_lb = con->lb->evaluate();
      con_ub = con->ub->evaluate();
      body_constant_val = con->constant_expr->evaluate();
      if (con_lb == con_ub)
	{
	  con_lb -= body_constant_val;
	  con_ub = con_lb;
	  f << con->name << "_eq: \n";
	  write_expr(f, con, false);
	  f << "= " << con_lb << " \n\n";
	}
      else if (con_lb > -inf && con_ub < inf)
	{
	  con_lb -= body_constant_val;
	  con_ub -= body_constant_val;
	  f << con->name << "_lb: \n";
	  write_expr(f, con, false);
	  f << ">= " << con_lb << " \n\n";
	  f << con->name << "_ub: \n";
	  write_expr(f, con, false);
	  f << "<= " << con_ub << " \n\n";
	}
      else if (con_lb > -inf)
	{
	  con_lb -= body_constant_val;
	  f << con->name << "_lb: \n";
	  write_expr(f, con, false);
	  f << ">= " << con_lb << " \n\n";
	}
      else if (con_ub < inf)
	{
	  con_ub -= body_constant_val;
	  f << con->name << "_ub: \n";
	  write_expr(f, con, false);
	  f << "<= " << con_ub << " \n\n";
	}
    }

  f << "obj_const_con_eq: \n";
  f << "+1 obj_const \n";
  f << "= " << objective->constant_expr->evaluate() << " \n\n";

  for (std::shared_ptr<LPConstraint> con : active_constraints)
    {
      for (std::shared_ptr<Var> v : *(con->linear_vars))
	{
	  v->index = -1;
	}
      for (std::shared_ptr<Var> v : *(con->quadratic_vars_1))
	{
	  v->index = -1;
	}
      for (std::shared_ptr<Var> v : *(con->quadratic_vars_2))
	{
	  v->index = -1;
	}
    }

  for (std::shared_ptr<Var> v : *(objective->linear_vars))
    {
      v->index = -1;
    }
  for (std::shared_ptr<Var> v : *(objective->quadratic_vars_1))
    {
      v->index = -1;
    }
  for (std::shared_ptr<Var> v : *(objective->quadratic_vars_2))
    {
      v->index = -1;
    }

  std::vector<std::shared_ptr<Var> > active_vars;
  for (std::shared_ptr<LPConstraint> con : active_constraints)
    {
      for (std::shared_ptr<Var> v : *(con->linear_vars))
	{
	  if (v->index == -1)
	    {
	      v->index = -2;
	      active_vars.push_back(v);
	    }
	}
      for (std::shared_ptr<Var> v : *(con->quadratic_vars_1))
	{
	  if (v->index == -1)
	    {
	      v->index = -2;
	      active_vars.push_back(v);
	    }
	}
      for (std::shared_ptr<Var> v : *(con->quadratic_vars_2))
	{
	  if (v->index == -1)
	    {
	      v->index = -2;
	      active_vars.push_back(v);
	    }
	}
    }

  for (std::shared_ptr<Var> v : *(objective->linear_vars))
    {
      if (v->index == -1)
	{
	  v->index = -2;
	  active_vars.push_back(v);
	}
    }
  for (std::shared_ptr<Var> v : *(objective->quadratic_vars_1))
    {
      if (v->index == -1)
	{
	  v->index = -2;
	  active_vars.push_back(v);
	}
    }
  for (std::shared_ptr<Var> v : *(objective->quadratic_vars_2))
    {
      if (v->index == -1)
	{
	  v->index = -2;
	  active_vars.push_back(v);
	}
    }

  f << "Bounds\n";
  std::vector<std::shared_ptr<Var> > binaries;
  std::vector<std::shared_ptr<Var> > integers;
  for (std::shared_ptr<Var> v : active_vars)
    {
      if (v->domain == "binary")
	{
	  binaries.push_back(v);
	}
      else if (v->domain == "integer")
	{
	  integers.push_back(v);
	}
      if (v->fixed)
	{
	  f << "  " << v->value << " <= " << v->name << " <= " << v->value << " \n";
	}
      else
	{
	  f << "  ";
	  if (v->lb <= -inf)
	    {
	      f << "-inf";
	    }
	  else
	    {
	      f << v->lb;
	    }
	  f << " <= " << v->name << " <= ";
	  if (v->ub >= inf)
	    {
	      f << "+inf";
	    }
	  else
	    {
	      f << v->ub;
	    }
	  f << " \n";
	}
    }
  f << "-inf <= obj_const <= +inf\n\n";

  if (binaries.size() > 0)
    {
      f << "Binaries \n";
      for (std::shared_ptr<Var> v : binaries)
	{
	  f << v->name << " \n";
	}
    }

  if (integers.size() > 0)
    {
      f << "Generals \n";
      for (std::shared_ptr<Var> v : integers)
	{
	  f << v->name << " \n";
	}
    }

  f << "end\n";
  
  f.close();

  solve_cons = active_constraints;
  solve_vars = active_vars;
}


std::vector<std::shared_ptr<Var> > LPWriter::get_solve_vars()
{
  return solve_vars;
}


std::vector<std::shared_ptr<LPConstraint> > LPWriter::get_solve_cons()
{
  return solve_cons;
}
