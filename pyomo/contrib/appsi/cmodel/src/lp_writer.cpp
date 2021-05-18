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


void process_lp_constraints(py::list cons, py::object writer)
{
  py::object generate_standard_repn = py::module_::import("pyomo.repn.standard_repn").attr("generate_standard_repn");
  py::object id = py::module_::import("pyomo.contrib.appsi.writers.lp_writer").attr("id");
  py::object ScalarParam = py::module_::import("pyomo.core.base.param").attr("ScalarParam");
  py::object _ParamData = py::module_::import("pyomo.core.base.param").attr("_ParamData");
  py::object NumericConstant = py::module_::import("pyomo.core.expr.numvalue").attr("NumericConstant");
  py::str cname;
  py::object repn;
  py::object getSymbol = writer.attr("_symbol_map").attr("getSymbol");
  py::object labeler = writer.attr("_con_labeler");
  py::object dfs_postorder_stack = writer.attr("_walker").attr("dfs_postorder_stack");
  LPWriter* c_writer = writer.attr("_writer").cast<LPWriter*>();
  py::dict var_map = writer.attr("_pyomo_var_to_solver_var_map");
  py::dict param_map = writer.attr("_pyomo_param_to_solver_param_map");
  py::dict pyomo_con_to_solver_con_map = writer.attr("_pyomo_con_to_solver_con_map");
  py::dict solver_con_to_pyomo_con_map = writer.attr("_solver_con_to_pyomo_con_map");
  py::int_ ione = 1;
  py::float_ fone = 1.0;
  py::type int_ = py::type::of(ione);
  py::type float_ = py::type::of(fone);
  py::type tmp_type = py::type::of(ione);
  std::shared_ptr<ExpressionBase> _const;
  py::object repn_constant;
  std::vector<std::shared_ptr<ExpressionBase> > lin_coef;
  std::vector<std::shared_ptr<Var> > lin_vars;
  py::list repn_linear_coefs;
  py::list repn_linear_vars;
  std::vector<std::shared_ptr<ExpressionBase> > quad_coef;
  std::vector<std::shared_ptr<Var> > quad_vars_1;
  std::vector<std::shared_ptr<Var> > quad_vars_2;
  py::list repn_quad_coefs;
  py::list repn_quad_vars;
  std::shared_ptr<LPConstraint> lp_con;
  py::tuple v_tuple;
  py::handle lb;
  py::handle ub;
  py::tuple lower_body_upper;
  py::dict active_constraints = writer.attr("_active_constraints");
  py::object nonlinear_expr;
  for (py::handle c : cons)
    {
      lower_body_upper = active_constraints[c];
      cname = getSymbol(c, labeler);
      repn = generate_standard_repn(lower_body_upper[1], "compute_values"_a=false, "quadratic"_a=true);
      nonlinear_expr = repn.attr("nonlinear_expr");
      if (!(nonlinear_expr.is(py::none())))
        {
          throw py::value_error("cannot write an LP file with a nonlinear constraint");
        }
      repn_constant = repn.attr("constant");
      tmp_type = py::type::of(repn_constant);
      if (tmp_type.is(int_) || tmp_type.is(float_))
        {
          _const = std::make_shared<Constant>(repn_constant.cast<double>());
        }
      else if(tmp_type.is(ScalarParam) || tmp_type.is(_ParamData))
        {
          _const = param_map[id(repn_constant)].cast<std::shared_ptr<ExpressionBase> >();
        }
      else
        {
          _const = dfs_postorder_stack(repn_constant).cast<std::shared_ptr<ExpressionBase> >();
        }
      lin_coef.clear();
      repn_linear_coefs = repn.attr("linear_coefs");
      for (py::handle coef : repn_linear_coefs)
        {
          tmp_type = py::type::of(coef);
          if (tmp_type.is(int_) || tmp_type.is(float_))
            {
              lin_coef.push_back(std::make_shared<Constant>(coef.cast<double>()));
            }
          else if(tmp_type.is(ScalarParam) || tmp_type.is(_ParamData))
            {
              lin_coef.push_back(param_map[id(coef)].cast<std::shared_ptr<ExpressionBase> >());
            }
          else
            {
              lin_coef.push_back(dfs_postorder_stack(coef).cast<std::shared_ptr<ExpressionBase> >());
            }
        }
      lin_vars.clear();
      repn_linear_vars = repn.attr("linear_vars");
      for (py::handle v : repn_linear_vars)
        {
          lin_vars.push_back(var_map[id(v)].cast<std::shared_ptr<Var> >());
        }
      quad_coef.clear();
      repn_quad_coefs = repn.attr("quadratic_coefs");
      for (py::handle coef : repn_quad_coefs)
        {
          tmp_type = py::type::of(coef);
          if (tmp_type.is(int_) || tmp_type.is(float_))
            {
              quad_coef.push_back(std::make_shared<Constant>(coef.cast<double>()));
            }
          else if(tmp_type.is(ScalarParam) || tmp_type.is(_ParamData))
            {
              quad_coef.push_back(param_map[id(coef)].cast<std::shared_ptr<ExpressionBase> >());
            }
          else
            {
              quad_coef.push_back(dfs_postorder_stack(coef).cast<std::shared_ptr<ExpressionBase> >());
            }
        }
      quad_vars_1.clear();
      quad_vars_2.clear();
      repn_quad_vars = repn.attr("quadratic_vars");
      for (py::handle v_tuple_handle : repn_quad_vars)
        {
          v_tuple = v_tuple_handle.cast<py::tuple>();
          quad_vars_1.push_back(var_map[id(v_tuple[0])].cast<std::shared_ptr<Var> >());
          quad_vars_2.push_back(var_map[id(v_tuple[1])].cast<std::shared_ptr<Var> >());
        }

      lp_con = std::make_shared<LPConstraint>(_const, lin_coef, lin_vars, quad_coef, quad_vars_1, quad_vars_2);
      lp_con->name = cname;

      lb = lower_body_upper[0];
      ub = lower_body_upper[2];
      if (!lb.is(py::none()))
        {
          tmp_type = py::type::of(lb);
          if (tmp_type.is(NumericConstant))
            {
              lp_con->lb = std::make_shared<Constant>(lb.attr("value").cast<double>());
            }
          else if (tmp_type.is(int_) || tmp_type.is(float_))
            {
              lp_con->lb = std::make_shared<Constant>(lb.cast<double>());
            }
          else if(tmp_type.is(ScalarParam) || tmp_type.is(_ParamData))
            {
              lp_con->lb = param_map[id(lb)].cast<std::shared_ptr<ExpressionBase> >();
            }
          else
            {
              lp_con->lb = dfs_postorder_stack(lb).cast<std::shared_ptr<ExpressionBase> >();
            }
        }
      if (!ub.is(py::none()))
        {
          tmp_type = py::type::of(ub);
          if (tmp_type.is(NumericConstant))
            {
              lp_con->ub = std::make_shared<Constant>(ub.attr("value").cast<double>());
            }
          else if (tmp_type.is(int_) || tmp_type.is(float_))
            {
              lp_con->ub = std::make_shared<Constant>(ub.cast<double>());
            }
          else if(tmp_type.is(ScalarParam) || tmp_type.is(_ParamData))
            {
              lp_con->ub = param_map[id(ub)].cast<std::shared_ptr<ExpressionBase> >();
            }
          else
            {
              lp_con->ub = dfs_postorder_stack(ub).cast<std::shared_ptr<ExpressionBase> >();
            }
        }
      c_writer->add_constraint(lp_con);
      pyomo_con_to_solver_con_map[c] = py::cast(lp_con);
      solver_con_to_pyomo_con_map[py::cast(lp_con)] = c;
    }
}
