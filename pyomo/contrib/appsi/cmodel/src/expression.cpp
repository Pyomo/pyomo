#include "expression.hpp"


std::shared_ptr<ExpressionBase> binary_helper(std::shared_ptr<ExpressionBase> n1, std::shared_ptr<ExpressionBase> n2, std::shared_ptr<BinaryOperator> oper)
{
  std::shared_ptr<Expression> expr = std::make_shared<Expression>();
  if (n1->is_leaf() && n2->is_leaf())
    {
      oper->operand1 = n1;
      oper->operand2 = n2;
    }
  else if (n1->is_leaf())
    {
      oper->operand1 = n1;
      std::shared_ptr<Expression> n2_expr = std::dynamic_pointer_cast<Expression>(n2);
      oper->operand2 = n2_expr->operators->back();
      if (n2_expr->operators->size() == n2_expr->n_operators)
	{
	  expr->operators = n2_expr->operators;
	}
      else
	{
	  for (unsigned int i=0; i<n2_expr->n_operators; ++i)
	    {
	      expr->operators->push_back(n2_expr->operators->at(i));
	    }
	}
    }
  else if (n2->is_leaf())
    {
      std::shared_ptr<Expression> n1_expr = std::dynamic_pointer_cast<Expression>(n1);
      oper->operand1 = n1_expr->operators->back();
      oper->operand2 = n2;
      if (n1_expr->operators->size() == n1_expr->n_operators)
	{
	  expr->operators = n1_expr->operators;
	}
      else
	{
	  for (unsigned int i=0; i<n1_expr->n_operators; ++i)
	    {
	      expr->operators->push_back(n1_expr->operators->at(i));
	    }
	}
    }
  else
    {
      std::shared_ptr<Expression> n1_expr = std::dynamic_pointer_cast<Expression>(n1);
      std::shared_ptr<Expression> n2_expr = std::dynamic_pointer_cast<Expression>(n2);
      oper->operand1 = n1_expr->operators->back();
      oper->operand2 = n2_expr->operators->back();
      if (n1_expr->operators->size() == n1_expr->n_operators)
	{
	  expr->operators = n1_expr->operators;
	}
      else
	{
	  for (unsigned int i=0; i<n1_expr->n_operators; ++i)
	    {
	      expr->operators->push_back(n1_expr->operators->at(i));
	    }
	}
      for (unsigned int i=0; i<n2_expr->n_operators; ++i)
	{
	  expr->operators->push_back(n2_expr->operators->at(i));
	}
    }
  expr->operators->push_back(oper);
  expr->n_operators = expr->operators->size();
  return expr;
}


std::shared_ptr<ExpressionBase> external_helper(std::string function_name, std::vector<std::shared_ptr<ExpressionBase> > operands)
{
  std::shared_ptr<ExternalOperator> oper = std::make_shared<ExternalOperator>();
  oper->function_name = function_name;
  std::shared_ptr<Expression> expr = std::make_shared<Expression>();
  for (std::shared_ptr<Node> n : operands)
    {
      if (n->is_leaf())
	{
	  oper->operands->push_back(n);
	}
      else
	{
	  std::shared_ptr<Expression> n_expr = std::dynamic_pointer_cast<Expression>(n);
	  oper->operands->push_back(n_expr->operators->back());
	  for (unsigned int i=0; i<n_expr->n_operators; ++i)
	    {
	      expr->operators->push_back(n_expr->operators->at(i));
	    }
	}
    }
  expr->operators->push_back(oper);
  expr->n_operators = expr->operators->size();
  return expr;
}


std::shared_ptr<ExpressionBase> unary_helper(std::shared_ptr<ExpressionBase> n1, std::shared_ptr<UnaryOperator> oper)
{
  std::shared_ptr<Expression> expr = std::make_shared<Expression>();
  if (n1->is_leaf())
    {
      oper->operand = n1;
    }
  else
    {
      std::shared_ptr<Expression> n1_expr = std::dynamic_pointer_cast<Expression>(n1);
      oper->operand = n1_expr->operators->back();
      if (n1_expr->operators->size() == n1_expr->n_operators)
	{
	  expr->operators = n1_expr->operators;
	}
      else
	{
	  for (unsigned int i=0; i<n1_expr->n_operators; ++i)
	    {
	      expr->operators->push_back(n1_expr->operators->at(i));
	    }
	}
    }
  expr->operators->push_back(oper);
  expr->n_operators = expr->operators->size();
  return expr;
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator+(ExpressionBase& other)
{
  if (other.is_constant_type() && other.evaluate() == 0.0)
    {
      return shared_from_this();
    }
  else if (is_constant_type() && evaluate() == 0.0)
    {
      return other.shared_from_this();
    }
  else
    {
      std::shared_ptr<AddOperator> oper = std::make_shared<AddOperator>();
      return binary_helper(shared_from_this(), other.shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator-(ExpressionBase& other)
{
  if (other.is_constant_type() && other.evaluate() == 0.0)
    {
      return shared_from_this();
    }
  else if (is_constant_type() && evaluate() == 0.0)
    {
      return -other;
    }
  else
    {
      std::shared_ptr<SubtractOperator> oper = std::make_shared<SubtractOperator>();
      return binary_helper(shared_from_this(), other.shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator*(ExpressionBase& other)
{
  if (other.is_constant_type() && other.evaluate() == 1.0)
    {
      return shared_from_this();
    }
  else if (other.is_constant_type() && other.evaluate() == 0.0)
    {
      return other.shared_from_this();
    }
  else if (is_constant_type() && evaluate() == 1.0)
    {
      return other.shared_from_this();
    }
  else if (is_constant_type() && evaluate() == 0.0)
    {
      return shared_from_this();
    }
  else
    {
      std::shared_ptr<MultiplyOperator> oper = std::make_shared<MultiplyOperator>();
      return binary_helper(shared_from_this(), other.shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator/(ExpressionBase& other)
{
  if (other.is_constant_type() && other.evaluate() == 1.0)
    {
      return shared_from_this();
    }
  else if (other.is_constant_type() && other.evaluate() == 0.0)
    {
      assert (false);
    }
  else if (is_constant_type() && evaluate() == 0.0)
    {
      return shared_from_this();
    }
  std::shared_ptr<DivideOperator> oper = std::make_shared<DivideOperator>();
  return binary_helper(shared_from_this(), other.shared_from_this(), oper);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__pow__(ExpressionBase& other)
{
  if (other.is_constant_type() && other.evaluate() == 1.0)
    {
      return shared_from_this();
    }
  else if (other.is_constant_type() && other.evaluate() == 0.0)
    {
      return std::make_shared<Constant>(1.0);
    }
  else if (is_constant_type() && evaluate() == 1.0)
    {
      return shared_from_this();
    }
  else if (is_constant_type() && evaluate() == 0.0)
    {
      return shared_from_this();
    }
  else
    {
      std::shared_ptr<PowerOperator> oper = std::make_shared<PowerOperator>();
      return binary_helper(shared_from_this(), other.shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator-()
{
  if (is_constant_type())
    {
      return std::make_shared<Constant>(-evaluate());
    }
  else
    {
      std::shared_ptr<NegationOperator> oper = std::make_shared<NegationOperator>();
      return unary_helper(shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> appsi_exp(std::shared_ptr<ExpressionBase> n)
{
  if (n->is_constant_type())
    {
      return std::make_shared<Constant>(std::exp(n->evaluate()));
    }
  else
    {
      std::shared_ptr<ExpOperator> oper = std::make_shared<ExpOperator>();
      return unary_helper(n->shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> appsi_log(std::shared_ptr<ExpressionBase> n)
{
  if (n->is_constant_type())
    {
      return std::make_shared<Constant>(std::log(n->evaluate()));
    }
  else
    {
      std::shared_ptr<LogOperator> oper = std::make_shared<LogOperator>();
      return unary_helper(n->shared_from_this(), oper);
    }
}


std::shared_ptr<ExpressionBase> appsi_sum(std::vector<std::shared_ptr<ExpressionBase> > exprs_to_sum)
{
  std::shared_ptr<Expression> res = std::make_shared<Expression>();
  std::shared_ptr<SumOperator> sum_op = std::make_shared<SumOperator>();
  for (std::shared_ptr<ExpressionBase> &e : exprs_to_sum)
    {
      if (e->is_leaf())
	{
	  sum_op->operands->push_back(e);
	}
      else
	{
	  std::shared_ptr<Expression> other = std::dynamic_pointer_cast<Expression>(e);
	  for (unsigned int i=0; i<other->n_operators; ++i)
	    {
	      res->operators->push_back(other->operators->at(i));
	    }
	  sum_op->operands->push_back(other->operators->at(other->n_operators - 1));
	}
    }
  res->operators->push_back(sum_op);
  res->n_operators = res->operators->size();
  return res;
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator+(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*this) + (*other_const);
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator*(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*this) * (*other_const);
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator-(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*this) - (*other_const);
}


std::shared_ptr<ExpressionBase> ExpressionBase::operator/(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*this) / (*other_const);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__pow__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return this->__pow__(*other_const);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__radd__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*other_const) + (*this);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__rmul__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*other_const) * (*this);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__rsub__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*other_const) - (*this);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__rdiv__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*other_const) / (*this);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__rtruediv__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return (*other_const) / (*this);
}


std::shared_ptr<ExpressionBase> ExpressionBase::__rpow__(double other)
{
  std::shared_ptr<Constant> other_const = std::make_shared<Constant>(other);
  return other_const->__pow__(*this);
}


bool ExternalOperator::is_external()
{
  return true;
}


bool Leaf::is_leaf()
{
  return true;
}


bool Var::is_variable_type()
{
  return true;
}


bool Param::is_param_type()
{
  return true;
}


bool Constant::is_constant_type()
{
  return true;
}


bool Expression::is_expression_type()
{
  return true;
}


double Leaf::evaluate()
{
  return value;
}


bool Operator::is_operator_type()
{
  return true;
}


double Leaf::get_value_from_array(double* val_array)
{
  return value;
}


double Expression::get_value_from_array(double* val_array)
{
  return val_array[n_operators-1];
}


double Operator::get_value_from_array(double* val_array)
{
  return val_array[index];
}


void MultiplyOperator::evaluate(double* values)
{
  values[index] = operand1->get_value_from_array(values) * operand2->get_value_from_array(values);
}


void ExternalOperator::evaluate(double* values)
{
  // It would be nice to implement this, but it will take some more work.
  // This would require dynamic linking to the external function.
  assert (false);
}


void AddOperator::evaluate(double* values)
{
  values[index] = operand1->get_value_from_array(values) + operand2->get_value_from_array(values);
}


void LinearOperator::evaluate(double* values)
{
  values[index] = 0.0;
  for (unsigned int i=0; i<variables->size(); ++i)
    {
      values[index] += coefficients->at(i)->evaluate() * variables->at(i)->evaluate();
    }
}


void SumOperator::evaluate(double* values)
{
  values[index] = 0.0;
  for (std::shared_ptr<Node> &n : *operands)
    {
      values[index] += n->get_value_from_array(values);
    }
}


void SubtractOperator::evaluate(double* values)
{
  values[index] = operand1->get_value_from_array(values) - operand2->get_value_from_array(values);
}


void DivideOperator::evaluate(double* values)
{
  values[index] = operand1->get_value_from_array(values) / operand2->get_value_from_array(values);
}


void PowerOperator::evaluate(double* values)
{
  values[index] = std::pow(operand1->get_value_from_array(values), operand2->get_value_from_array(values));
}


void NegationOperator::evaluate(double* values)
{
  values[index] = -operand->get_value_from_array(values);
}


void ExpOperator::evaluate(double* values)
{
  values[index] = std::exp(operand->get_value_from_array(values));
}


void LogOperator::evaluate(double* values)
{
  values[index] = std::log(operand->get_value_from_array(values));
}


double Expression::evaluate()
{
  double* values = new double[n_operators];
  for (unsigned int i=0; i<n_operators; ++i)
    {
      operators->at(i)->index = i;
      operators->at(i)->evaluate(values);
    }
  double res = get_value_from_array(values);
  delete[] values;
  return res;
}


void UnaryOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  if (operand->is_variable_type())
    {
      var_set.insert(operand);
    }
}


void BinaryOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  if (operand1->is_variable_type())
    {
      var_set.insert(operand1);
    }
  if (operand2->is_variable_type())
    {
      var_set.insert(operand2);
    }
}


void ExternalOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  for (unsigned int i=0; i<operands->size(); ++i)
    {
      if (operands->at(i)->is_variable_type())
	{
	  var_set.insert(operands->at(i));
	}
    }
}


void LinearOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  for (std::shared_ptr<Var> v : *variables)
    {
      var_set.insert(v);
    }
}


void SumOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  for (std::shared_ptr<Node> &n : *operands)
    {
      if (n->is_variable_type())
	{
	  var_set.insert(n);
	}
    }
}


std::shared_ptr<std::vector<std::shared_ptr<Var> > > Expression::identify_variables()
{
  std::set<std::shared_ptr<Node> > var_set;
  for (unsigned int i=0; i<n_operators; ++i)
    {
      operators->at(i)->identify_variables(var_set);
    }
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > res = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  for (std::shared_ptr<Node> v : var_set)
    {
      res->push_back(std::dynamic_pointer_cast<Var>(v));
    }
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<Var> > > Var::identify_variables()
{
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > res = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  res->push_back(shared_from_this());
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<Var> > > Constant::identify_variables()
{
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > res = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<Var> > > Param::identify_variables()
{
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > res = std::make_shared<std::vector<std::shared_ptr<Var> > >();
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > Expression::identify_external_operators()
{
  std::set<std::shared_ptr<Node> > external_set;
  for (unsigned int i=0; i<n_operators; ++i)
    {
      if (operators->at(i)->is_external())
	{
	  external_set.insert(operators->at(i));
	}
    }
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > res = std::make_shared<std::vector<std::shared_ptr<ExternalOperator> > >();
  for (std::shared_ptr<Node> n : external_set)
    {
      res->push_back(std::dynamic_pointer_cast<ExternalOperator>(n));
    }
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > Var::identify_external_operators()
{
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > res = std::make_shared<std::vector<std::shared_ptr<ExternalOperator> > >();
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > Constant::identify_external_operators()
{
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > res = std::make_shared<std::vector<std::shared_ptr<ExternalOperator> > >();
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > Param::identify_external_operators()
{
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > res = std::make_shared<std::vector<std::shared_ptr<ExternalOperator> > >();
  return res;
}




int Var::get_degree_from_array(int* degree_array)
{
  return 1;
}


int Param::get_degree_from_array(int* degree_array)
{
  return 0;
}


int Constant::get_degree_from_array(int* degree_array)
{
  return 0;
}


int Expression::get_degree_from_array(int* degree_array)
{
  return degree_array[n_operators-1];
}


int Operator::get_degree_from_array(int* degree_array)
{
  return degree_array[index];
}


bool Leaf::get_accounted_for_from_array(bool* accounted_for)
{
  return false;
}


bool Expression::get_accounted_for_from_array(bool* accounted_for)
{
  return accounted_for[n_operators-1];
}


bool Operator::get_accounted_for_from_array(bool* accounted_for)
{
  return accounted_for[index];
}


void Leaf::set_repn_info(bool* _array, bool _value)
{
  ;
}


void Leaf::set_repn_info(int* _array, int _value)
{
  ;
}


void Expression::set_repn_info(bool* _array, bool _value)
{
  _array[n_operators-1] = _value;
}


void Expression::set_repn_info(int* _array, int _value)
{
  _array[n_operators-1] = _value;
}


void Operator::set_repn_info(bool* _array, bool _value)
{
  _array[index] = _value;
}


void Operator::set_repn_info(int* _array, int _value)
{
  _array[index] = _value;
}


void LinearOperator::propagate_degree_forward(int* degrees, double* values)
{
  degrees[index] = 1;
}


void SumOperator::propagate_degree_forward(int* degrees, double* values)
{
  int deg = 0;
  int _deg;
  for (std::shared_ptr<Node> &n : *operands)
    {
      _deg = n->get_degree_from_array(degrees);
      if (_deg > deg)
	{
	  deg = _deg;
	}
    }
  degrees[index] = deg;
}


void MultiplyOperator::propagate_degree_forward(int* degrees, double* values)
{
  degrees[index] = operand1->get_degree_from_array(degrees) + operand2->get_degree_from_array(degrees);
}


void ExternalOperator::propagate_degree_forward(int* degrees, double* values)
{
  // External functions are always considered nonlinear
  // Anything larger than 2 is nonlinear
  degrees[index] = 3;
}


void AddOperator::propagate_degree_forward(int* degrees, double* values)
{
  degrees[index] = std::max(operand1->get_degree_from_array(degrees), operand2->get_degree_from_array(degrees));
}


void SubtractOperator::propagate_degree_forward(int* degrees, double* values)
{
  degrees[index] = std::max(operand1->get_degree_from_array(degrees), operand2->get_degree_from_array(degrees));
}


void DivideOperator::propagate_degree_forward(int* degrees, double* values)
{
  // anything larger than 2 is nonlinear
  degrees[index] = std::max(operand1->get_degree_from_array(degrees), 3*(operand2->get_degree_from_array(degrees)));
}


void PowerOperator::propagate_degree_forward(int* degrees, double* values)
{
  if (operand2->get_degree_from_array(degrees) != 0)
    {
      degrees[index] = 3;
    }
  else
    {
      double val2 = operand2->get_value_from_array(values);
      double intpart;
      if (std::modf(val2, &intpart) == 0.0)
	{
	  degrees[index] = operand1->get_degree_from_array(degrees) * (int)val2;
	}
      else
	{
	  degrees[index] = 3;
	}
    }
}


void NegationOperator::propagate_degree_forward(int* degrees, double* values)
{
  degrees[index] = operand->get_degree_from_array(degrees);
}


void ExpOperator::propagate_degree_forward(int* degrees, double* values)
{
  if (operand->get_degree_from_array(degrees) == 0)
    {
      degrees[index] = 0;
    }
  else
    {
      degrees[index] = 3;
    }
}


void LogOperator::propagate_degree_forward(int* degrees, double* values)
{
  if (operand->get_degree_from_array(degrees) == 0)
    {
      degrees[index] = 0;
    }
  else
    {
      degrees[index] = 3;
    }
}


void AddOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  if (push[index])
    {
      operand1->set_repn_info(degrees, degrees[index]);
      operand1->set_repn_info(push, true);
      operand2->set_repn_info(degrees, degrees[index]);
      operand2->set_repn_info(push, true);
    }
  operand1->set_repn_info(negate, negate[index]);
  operand2->set_repn_info(negate, negate[index]);
}


void SubtractOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  if (push[index])
    {
      operand1->set_repn_info(degrees, degrees[index]);
      operand1->set_repn_info(push, true);
      operand2->set_repn_info(degrees, degrees[index]);
      operand2->set_repn_info(push, true);
    }
  operand1->set_repn_info(negate, negate[index]);
  operand2->set_repn_info(negate, !negate[index]);
}


void MultiplyOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  operand1->set_repn_info(degrees, degrees[index]);
  operand1->set_repn_info(push, true);
  operand2->set_repn_info(degrees, degrees[index]);
  operand2->set_repn_info(push, true);
}


void DivideOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  operand1->set_repn_info(degrees, degrees[index]);
  operand1->set_repn_info(push, true);
  operand2->set_repn_info(degrees, degrees[index]);
  operand2->set_repn_info(push, true);
}


void PowerOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  operand1->set_repn_info(degrees, degrees[index]);
  operand1->set_repn_info(push, true);
  operand2->set_repn_info(degrees, degrees[index]);
  operand2->set_repn_info(push, true);
}


void NegationOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  if (push[index])
    {
      operand->set_repn_info(degrees, degrees[index]);
      operand->set_repn_info(push, true);
    }
  operand->set_repn_info(negate, !negate[index]);
}


void ExpOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  operand->set_repn_info(degrees, degrees[index]);
  operand->set_repn_info(push, true);
}


void LogOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  operand->set_repn_info(degrees, degrees[index]);
  operand->set_repn_info(push, true);
}


void ExternalOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  for (std::shared_ptr<Node> &operand : (*operands))
    {
      operand->set_repn_info(degrees, degrees[index]);
      operand->set_repn_info(push, true);
    }
}


void SumOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  for (std::shared_ptr<Node> &n : *operands)
    {
      if (push[index])
	{
	  n->set_repn_info(degrees, degrees[index]);
	  n->set_repn_info(push, true);
	}
      n->set_repn_info(negate, negate[index]);
    }
}


void LinearOperator::propagate_repn_info_backward(int* degrees, bool* push, bool* negate)
{
  ;
}


void Repn::reset_with_constants()
{
  constant = std::make_shared<Constant>();
  linear = std::make_shared<Constant>();
  quadratic = std::make_shared<Constant>();
  nonlinear = std::make_shared<Constant>();
}


void TmpRepn::reset_with_expressions()
{
  constant = std::make_shared<Expression>();
  linear = std::make_shared<Expression>();
  quadratic = std::make_shared<Expression>();
  nonlinear = std::make_shared<Expression>();
}


void AddOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				int* degrees,
				bool* push,
				bool* accounted_for,
				bool* negate,
				std::shared_ptr<SumOperator> constant_sum, 
				std::shared_ptr<SumOperator> linear_sum, 
				std::shared_ptr<SumOperator> quadratic_sum, 
				std::shared_ptr<SumOperator> nonlinear_sum)
{
  if (push[index])
    {
      accounted_for[index] = false;
      int deg = degrees[index];
      if (deg == 0)
	{
	  tmp_repn->constant->operators->push_back(shared_from_this());
	}
      else if (deg == 1)
	{
	  tmp_repn->linear->operators->push_back(shared_from_this());
	}
      else if (deg == 2)
	{
	  tmp_repn->quadratic->operators->push_back(shared_from_this());
	}
      else
	{
	  tmp_repn->nonlinear->operators->push_back(shared_from_this());
	}      
    }
  else
    {
      accounted_for[index] = true;
      bool accounted_for1 = operand1->get_accounted_for_from_array(accounted_for);
      bool accounted_for2 = operand2->get_accounted_for_from_array(accounted_for);

      if (negate[index])
	{
	  if (!accounted_for1)
	    {
	      int deg1 = operand1->get_degree_from_array(degrees);
	      std::shared_ptr<NegationOperator> neg = std::make_shared<NegationOperator>();
	      neg->operand = operand1;
	      if (deg1 == 0)
		{
		  tmp_repn->constant->operators->push_back(neg);
		  constant_sum->operands->push_back(neg);
		}
	      else if (deg1 == 1)
		{
		  tmp_repn->linear->operators->push_back(neg);
		  linear_sum->operands->push_back(neg);
		}
	      else if (deg1 == 2)
		{
		  tmp_repn->quadratic->operators->push_back(neg);
		  quadratic_sum->operands->push_back(neg);
		}
	      else
		{
		  tmp_repn->nonlinear->operators->push_back(neg);
		  nonlinear_sum->operands->push_back(neg);
		}
	    }
	  if (!accounted_for2)
	    {
	      int deg2 = operand2->get_degree_from_array(degrees);
	      std::shared_ptr<NegationOperator> neg = std::make_shared<NegationOperator>();
	      neg->operand = operand2;
	      if (deg2 == 0)
		{
		  tmp_repn->constant->operators->push_back(neg);
		  constant_sum->operands->push_back(neg);
		}
	      else if (deg2 == 1)
		{
		  tmp_repn->linear->operators->push_back(neg);
		  linear_sum->operands->push_back(neg);
		}
	      else if (deg2 == 2)
		{
		  tmp_repn->quadratic->operators->push_back(neg);
		  quadratic_sum->operands->push_back(neg);
		}
	      else
		{
		  tmp_repn->nonlinear->operators->push_back(neg);
		  nonlinear_sum->operands->push_back(neg);
		}
	    }
	}
      else
	{
	  if (!accounted_for1)
	    {
	      int deg1 = operand1->get_degree_from_array(degrees);
	      if (deg1 == 0)
		{
		  constant_sum->operands->push_back(operand1);
		}
	      else if (deg1 == 1)
		{
		  linear_sum->operands->push_back(operand1);
		}
	      else if (deg1 == 2)
		{
		  quadratic_sum->operands->push_back(operand1);
		}
	      else
		{
		  nonlinear_sum->operands->push_back(operand1);
		}
	    }
	  
	  if (!accounted_for2)
	    {
	      int deg2 = operand2->get_degree_from_array(degrees);
	      if (deg2 == 0)
		{
		  constant_sum->operands->push_back(operand2);
		}
	      else if (deg2 == 1)
		{
		  linear_sum->operands->push_back(operand2);
		}
	      else if (deg2 == 2)
		{
		  quadratic_sum->operands->push_back(operand2);
		}
	      else
		{
		  nonlinear_sum->operands->push_back(operand2);
		}
	    }
	}
    }
}


void SubtractOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				     int* degrees,
				     bool* push,
				     bool* accounted_for,
				     bool* negate,
				     std::shared_ptr<SumOperator> constant_sum, 
				     std::shared_ptr<SumOperator> linear_sum, 
				     std::shared_ptr<SumOperator> quadratic_sum, 
				     std::shared_ptr<SumOperator> nonlinear_sum)
{
  if (push[index])
    {
      accounted_for[index] = false;
      int deg = degrees[index];
      if (deg == 0)
	{
	  tmp_repn->constant->operators->push_back(shared_from_this());
	}
      else if (deg == 1)
	{
	  tmp_repn->linear->operators->push_back(shared_from_this());
	}
      else if (deg == 2)
	{
	  tmp_repn->quadratic->operators->push_back(shared_from_this());
	}
      else
	{
	  tmp_repn->nonlinear->operators->push_back(shared_from_this());
	}      
    }
  else
    {
      accounted_for[index] = true;
      bool accounted_for1 = operand1->get_accounted_for_from_array(accounted_for);
      bool accounted_for2 = operand2->get_accounted_for_from_array(accounted_for);

      if (negate[index])
	{
	  if (!accounted_for1)
	    {
	      int deg1 = operand1->get_degree_from_array(degrees);
	      std::shared_ptr<NegationOperator> neg = std::make_shared<NegationOperator>();
	      neg->operand = operand1;
	      if (deg1 == 0)
		{
		  tmp_repn->constant->operators->push_back(neg);
		  constant_sum->operands->push_back(neg);
		}
	      else if (deg1 == 1)
		{
		  tmp_repn->linear->operators->push_back(neg);
		  linear_sum->operands->push_back(neg);
		}
	      else if (deg1 == 2)
		{
		  tmp_repn->quadratic->operators->push_back(neg);
		  quadratic_sum->operands->push_back(neg);
		}
	      else
		{
		  tmp_repn->nonlinear->operators->push_back(neg);
		  nonlinear_sum->operands->push_back(neg);
		}
	    }

	  if (!accounted_for2)
	    {
	      int deg2 = operand2->get_degree_from_array(degrees);
	      if (deg2 == 0)
		{
		  constant_sum->operands->push_back(operand2);
		}
	      else if (deg2 == 1)
		{
		  linear_sum->operands->push_back(operand2);
		}
	      else if (deg2 == 2)
		{
		  quadratic_sum->operands->push_back(operand2);
		}
	      else
		{
		  nonlinear_sum->operands->push_back(operand2);
		}
	    }
	}
      else
	{
	  if (!accounted_for1)
	    {
	      int deg1 = operand1->get_degree_from_array(degrees);
	      if (deg1 == 0)
		{
		  constant_sum->operands->push_back(operand1);
		}
	      else if (deg1 == 1)
		{
		  linear_sum->operands->push_back(operand1);
		}
	      else if (deg1 == 2)
		{
		  quadratic_sum->operands->push_back(operand1);
		}
	      else
		{
		  nonlinear_sum->operands->push_back(operand1);
		}
	    }
	  
	  if (!accounted_for2)
	    {
	      int deg2 = operand2->get_degree_from_array(degrees);
	      std::shared_ptr<NegationOperator> neg = std::make_shared<NegationOperator>();
	      neg->operand = operand2;
	      if (deg2 == 0)
		{
		  tmp_repn->constant->operators->push_back(neg);
		  constant_sum->operands->push_back(neg);
		}
	      else if (deg2 == 1)
		{
		  tmp_repn->linear->operators->push_back(neg);
		  linear_sum->operands->push_back(neg);
		}
	      else if (deg2 == 2)
		{
		  tmp_repn->quadratic->operators->push_back(neg);
		  quadratic_sum->operands->push_back(neg);
		}
	      else
		{
		  tmp_repn->nonlinear->operators->push_back(neg);
		  nonlinear_sum->operands->push_back(neg);
		}
	    }
	}
    }
}


void SumOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				int* degrees,
				bool* push,
				bool* accounted_for,
				bool* negate,
				std::shared_ptr<SumOperator> constant_sum, 
				std::shared_ptr<SumOperator> linear_sum, 
				std::shared_ptr<SumOperator> quadratic_sum, 
				std::shared_ptr<SumOperator> nonlinear_sum)
{
  if (push[index])
    {
      accounted_for[index] = false;
      int deg = degrees[index];
      if (deg == 0)
	{
	  tmp_repn->constant->operators->push_back(shared_from_this());
	}
      else if (deg == 1)
	{
	  tmp_repn->linear->operators->push_back(shared_from_this());
	}
      else if (deg == 2)
	{
	  tmp_repn->quadratic->operators->push_back(shared_from_this());
	}
      else
	{
	  tmp_repn->nonlinear->operators->push_back(shared_from_this());
	}      
    }
  else
    {
      accounted_for[index] = true;
      bool operand_accounted_for;
      int deg;
      if (negate[index])
	{
	  for (std::shared_ptr<Node> &n : *operands)
	    {
	      operand_accounted_for = n->get_accounted_for_from_array(accounted_for);
	      if (!operand_accounted_for)
		{
		  deg = n->get_degree_from_array(degrees);
		  std::shared_ptr<NegationOperator> neg = std::make_shared<NegationOperator>();
		  neg->operand = n;
		  if (deg == 0)
		    {
		      tmp_repn->constant->operators->push_back(neg);
		      constant_sum->operands->push_back(neg);
		    }
		  else if (deg == 1)
		    {
		      tmp_repn->linear->operators->push_back(neg);
		      linear_sum->operands->push_back(neg);
		    }
		  else if (deg == 2)
		    {
		      tmp_repn->quadratic->operators->push_back(neg);
		      quadratic_sum->operands->push_back(neg);
		    }
		  else
		    {
		      tmp_repn->nonlinear->operators->push_back(neg);
		      nonlinear_sum->operands->push_back(neg);
		    }
		}
	    }
	}
      else
	{
	  for (std::shared_ptr<Node> &n : *operands)
	    {
	      operand_accounted_for = n->get_accounted_for_from_array(accounted_for);
	      if (!operand_accounted_for)
		{
		  deg = n->get_degree_from_array(degrees);
		  if (deg == 0)
		    {
		      constant_sum->operands->push_back(n);
		    }
		  else if (deg == 1)
		    {
		      linear_sum->operands->push_back(n);
		    }
		  else if (deg == 2)
		    {
		      quadratic_sum->operands->push_back(n);
		    }
		  else
		    {
		      nonlinear_sum->operands->push_back(n);
		    }
		}
	    }
	}
    }
}


void MultiplyOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				     int* degrees,
				     bool* push,
				     bool* accounted_for,
				     bool* negate,
				     std::shared_ptr<SumOperator> constant_sum, 
				     std::shared_ptr<SumOperator> linear_sum, 
				     std::shared_ptr<SumOperator> quadratic_sum, 
				     std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else if (deg == 1)
    {
      tmp_repn->linear->operators->push_back(shared_from_this());
    }
  else if (deg == 2)
    {
      tmp_repn->quadratic->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void DivideOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				   int* degrees,
				   bool* push,
				   bool* accounted_for,
				   bool* negate,
				   std::shared_ptr<SumOperator> constant_sum, 
				   std::shared_ptr<SumOperator> linear_sum, 
				   std::shared_ptr<SumOperator> quadratic_sum, 
				   std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else if (deg == 1)
    {
      tmp_repn->linear->operators->push_back(shared_from_this());
    }
  else if (deg == 2)
    {
      tmp_repn->quadratic->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void PowerOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				  int* degrees,
				  bool* push,
				  bool* accounted_for,
				  bool* negate,
				  std::shared_ptr<SumOperator> constant_sum, 
				  std::shared_ptr<SumOperator> linear_sum, 
				  std::shared_ptr<SumOperator> quadratic_sum, 
				  std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else if (deg == 1)
    {
      tmp_repn->linear->operators->push_back(shared_from_this());
    }
  else if (deg == 2)
    {
      tmp_repn->quadratic->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void NegationOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				     int* degrees,
				     bool* push,
				     bool* accounted_for,
				     bool* negate,
				     std::shared_ptr<SumOperator> constant_sum, 
				     std::shared_ptr<SumOperator> linear_sum, 
				     std::shared_ptr<SumOperator> quadratic_sum, 
				     std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else if (deg == 1)
    {
      tmp_repn->linear->operators->push_back(shared_from_this());
    }
  else if (deg == 2)
    {
      tmp_repn->quadratic->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void ExpOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				int* degrees,
				bool* push,
				bool* accounted_for,
				bool* negate,
				std::shared_ptr<SumOperator> constant_sum, 
				std::shared_ptr<SumOperator> linear_sum, 
				std::shared_ptr<SumOperator> quadratic_sum, 
				std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void LogOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				int* degrees,
				bool* push,
				bool* accounted_for,
				bool* negate,
				std::shared_ptr<SumOperator> constant_sum, 
				std::shared_ptr<SumOperator> linear_sum, 
				std::shared_ptr<SumOperator> quadratic_sum, 
				std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void ExternalOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				     int* degrees,
				     bool* push,
				     bool* accounted_for,
				     bool* negate,
				     std::shared_ptr<SumOperator> constant_sum, 
				     std::shared_ptr<SumOperator> linear_sum, 
				     std::shared_ptr<SumOperator> quadratic_sum, 
				     std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


void LinearOperator::generate_repn(std::shared_ptr<TmpRepn> tmp_repn,
				   int* degrees,
				   bool* push,
				   bool* accounted_for,
				   bool* negate,
				   std::shared_ptr<SumOperator> constant_sum, 
				   std::shared_ptr<SumOperator> linear_sum, 
				   std::shared_ptr<SumOperator> quadratic_sum, 
				   std::shared_ptr<SumOperator> nonlinear_sum)
{
  accounted_for[index] = false;
  int deg = degrees[index];
  if (deg == 0)
    {
      tmp_repn->constant->operators->push_back(shared_from_this());
    }
  else if (deg == 1)
    {
      tmp_repn->linear->operators->push_back(shared_from_this());
    }
  else if (deg == 2)
    {
      tmp_repn->quadratic->operators->push_back(shared_from_this());
    }
  else
    {
      tmp_repn->nonlinear->operators->push_back(shared_from_this());
    }
}


std::shared_ptr<Repn> Expression::generate_repn()
{
  int degrees[n_operators];
  bool push[n_operators];
  double values[n_operators];
  bool negate[n_operators];
  std::shared_ptr<Operator> oper;
  for (unsigned int i=0; i<n_operators; ++i)
    {
      push[i] = false;
      negate[i] = false;
      oper = operators->at(i);
      oper->index = i;
      oper->evaluate(values);
      oper->propagate_degree_forward(degrees, values);
    }
  for (int i=n_operators-1; i>=0; --i)
    {
      operators->at(i)->propagate_repn_info_backward(degrees, push, negate);
    }

  std::shared_ptr<Repn> res = std::make_shared<Repn>();
  std::shared_ptr<TmpRepn> tmp = std::make_shared<TmpRepn>();
  std::shared_ptr<SumOperator> constant_sum = std::make_shared<SumOperator>();
  std::shared_ptr<SumOperator> linear_sum = std::make_shared<SumOperator>();
  std::shared_ptr<SumOperator> quadratic_sum = std::make_shared<SumOperator>();
  std::shared_ptr<SumOperator> nonlinear_sum = std::make_shared<SumOperator>();

  res->reset_with_constants();
  tmp->reset_with_expressions();

  bool accounted_for[n_operators];
  for (unsigned int i=0; i<n_operators; ++i)
    {
      operators->at(i)->generate_repn(tmp,
				      degrees,
				      push,
				      accounted_for,
				      negate,
				      constant_sum,
				      linear_sum,
				      quadratic_sum,
				      nonlinear_sum);
    }

  if (constant_sum->operands->size() > 0)
    {
      tmp->constant->operators->push_back(constant_sum);
    }
  if (linear_sum->operands->size() > 0)
    {
      tmp->linear->operators->push_back(linear_sum);
    }
  if (quadratic_sum->operands->size() > 0)
    {
      tmp->quadratic->operators->push_back(quadratic_sum);
    }
  if (nonlinear_sum->operands->size() > 0)
    {
      tmp->nonlinear->operators->push_back(nonlinear_sum);
    }

  if (tmp->constant->operators->size() > 0)
    {
      tmp->constant->n_operators = tmp->constant->operators->size();
      res->constant = tmp->constant;
    }
  if (tmp->linear->operators->size() > 0)
    {
      tmp->linear->n_operators = tmp->linear->operators->size();
      res->linear = tmp->linear;
    }
  if (tmp->quadratic->operators->size() > 0)
    {
      tmp->quadratic->n_operators = tmp->quadratic->operators->size();
      res->quadratic = tmp->quadratic;
    }
  if (tmp->nonlinear->operators->size() > 0)
    {
      tmp->nonlinear->n_operators = tmp->nonlinear->operators->size();
      res->nonlinear = tmp->nonlinear;
    }

  return res;
}


std::shared_ptr<Repn> Var::generate_repn()
{
  std::shared_ptr<Repn> res = std::make_shared<Repn>();
  res->reset_with_constants();
  res->linear = shared_from_this();
  return res;
}


std::shared_ptr<Repn> Constant::generate_repn()
{
  std::shared_ptr<Repn> res = std::make_shared<Repn>();
  res->reset_with_constants();
  res->constant = shared_from_this();
  return res;
}


std::shared_ptr<Repn> Param::generate_repn()
{
  std::shared_ptr<Repn> res = std::make_shared<Repn>();
  res->reset_with_constants();
  res->constant = shared_from_this();
  return res;
}


std::string Var::__str__()
{
  return name;
}


std::string Param::__str__()
{
  return name;
}


std::string Constant::__str__()
{
  return std::to_string(value);
}


std::string Expression::__str__()
{
  std::string* string_array = new std::string[n_operators];
  std::shared_ptr<Operator> oper;
  for (unsigned int i=0; i<n_operators; ++i)
    {
      oper = operators->at(i);
      oper->index = i;
      oper->print(string_array);
    }
  std::string res = string_array[n_operators-1];
  delete[] string_array;
  return res;
}


std::string Leaf::get_string_from_array(std::string* string_array)
{
  return __str__();
}


std::string Expression::get_string_from_array(std::string* string_array)
{
  return string_array[n_operators-1];
}


std::string Operator::get_string_from_array(std::string* string_array)
{
  return string_array[index];
}


void AddOperator::print(std::string* string_array)
{
  string_array[index] = ("(" +
			 operand1->get_string_from_array(string_array) +
			 " + " +
			 operand2->get_string_from_array(string_array) +
			 ")");
}


void SubtractOperator::print(std::string* string_array)
{
  string_array[index] = ("(" +
			 operand1->get_string_from_array(string_array) +
			 " - " +
			 operand2->get_string_from_array(string_array) +
			 ")");
}


void MultiplyOperator::print(std::string* string_array)
{
  string_array[index] = ("(" +
			 operand1->get_string_from_array(string_array) +
			 "*" +
			 operand2->get_string_from_array(string_array) +
			 ")");
}


void ExternalOperator::print(std::string* string_array)
{
  std::string res = function_name + "(";
  for (unsigned int i=0; i<(operands->size() - 1); ++i)
    {
      res += operands->at(i)->get_string_from_array(string_array);
      res += ", ";
    }
  res += operands->back()->get_string_from_array(string_array);
  res += ")";
  string_array[index] = res;
}


void DivideOperator::print(std::string* string_array)
{
  string_array[index] = ("(" +
			 operand1->get_string_from_array(string_array) +
			 "/" +
			 operand2->get_string_from_array(string_array) +
			 ")");
}


void PowerOperator::print(std::string* string_array)
{
  string_array[index] = ("(" +
			 operand1->get_string_from_array(string_array) +
			 "**" +
			 operand2->get_string_from_array(string_array) +
			 ")");
}


void NegationOperator::print(std::string* string_array)
{
  string_array[index] = ("(-" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void ExpOperator::print(std::string* string_array)
{
  string_array[index] = ("exp(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void LogOperator::print(std::string* string_array)
{
  string_array[index] = ("log(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void LinearOperator::print(std::string* string_array)
{
  std::string res = "(" + coefficients->at(0)->__str__() + "*" + variables->at(0)->__str__();
  for (unsigned int i=1; i<variables->size(); ++i)
    {
      res += " + " + coefficients->at(i)->__str__() + "*" + variables->at(i)->__str__();
    }
  res += ")";
  string_array[index] = res;
}


void SumOperator::print(std::string* string_array)
{
  std::string res = "(" + operands->at(0)->get_string_from_array(string_array);
  for (unsigned int i=1; i<operands->size(); ++ i)
    {
      res += " + " + operands->at(i)->get_string_from_array(string_array);
    }
  res += ")";
  string_array[index] = res;
}


std::shared_ptr<std::vector<std::shared_ptr<Node> > > Leaf::get_prefix_notation()
{
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > res = std::make_shared<std::vector<std::shared_ptr<Node> > >();
  res->push_back(shared_from_this());
  return res;
}


std::shared_ptr<std::vector<std::shared_ptr<Node> > > Expression::get_prefix_notation()
{
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > res = std::make_shared<std::vector<std::shared_ptr<Node> > >();
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack = std::make_shared<std::vector<std::shared_ptr<Node> > >();
  std::shared_ptr<Node> node;
  stack->push_back(operators->at(n_operators - 1));
  while (stack->size() > 0)
    {
      node = stack->back();
      stack->pop_back();
      res->push_back(node);
      node->fill_prefix_notation_stack(stack);
    }
  
  return res;
}


void BinaryOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  stack->push_back(operand2);
  stack->push_back(operand1);
}


void UnaryOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  stack->push_back(operand);
}


void SumOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  int ndx = operands->size() - 1;
  while (ndx >= 0)
    {
      stack->push_back(operands->at(ndx));
      ndx -= 1;
    }
}


void LinearOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  throw std::string("This should not be encountered");
}


void ExternalOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  int i = operands->size() - 1;
  while (i >= 0)
    {
      stack->push_back(operands->at(i));
      i -= 1;
    }
}


void Var::write_nl_string(std::ofstream& f)
{
  f << "v" << index << "\n";
}


void Param::write_nl_string(std::ofstream& f)
{
  f << "n" << value << "\n";
}


void Constant::write_nl_string(std::ofstream& f)
{
  f << "n" << value << "\n";
}


void Expression::write_nl_string(std::ofstream& f)
{
  assert (false);
}


void MultiplyOperator::write_nl_string(std::ofstream& f)
{
  f << "o2\n";
}


void ExternalOperator::write_nl_string(std::ofstream& f)
{
  f << "f" << external_function_index << " " << operands->size() << "\n";
}


void SumOperator::write_nl_string(std::ofstream& f)
{
  if (operands->size() == 2)
    {
      f << "o0\n";
    }
  else
    {
      f << "o54\n";
      f << operands->size() << "\n";
    }
}


void LinearOperator::write_nl_string(std::ofstream& f)
{
  throw std::string("This should not be encountered");
}


void AddOperator::write_nl_string(std::ofstream& f)
{
  f << "o0\n";
}


void SubtractOperator::write_nl_string(std::ofstream& f)
{
  f << "o1\n";
}


void DivideOperator::write_nl_string(std::ofstream& f)
{
  f << "o3\n";
}


void PowerOperator::write_nl_string(std::ofstream& f)
{
  f << "o5\n";
}


void NegationOperator::write_nl_string(std::ofstream& f)
{
  f << "o16\n";
}


void ExpOperator::write_nl_string(std::ofstream& f)
{
  f << "o44\n";
}


void LogOperator::write_nl_string(std::ofstream& f)
{
  f << "o43\n";
}


std::shared_ptr<Operator> LinearOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  // this should never do anything
  return shared_from_this();
}


std::shared_ptr<Operator> SumOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_any_op = false;
  bool replace_op;

  for (std::shared_ptr<Node> &op : (*operands))
    {
      replace_op = needs_replaced->count(op);
      replace_any_op = replace_any_op || replace_op;
    }

  if (replace_any_op)
    {
      std::shared_ptr<SumOperator> new_op = std::make_shared<SumOperator>();
      for (std::shared_ptr<Node> &op : (*operands))
	{
	  replace_op = needs_replaced->count(op);
	  if (replace_op)
	    {
	      new_op->operands->push_back(needs_replaced->at(op));
	    }
	  else
	    {
	      new_op->operands->push_back(op);
	    }
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> MultiplyOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op1 = needs_replaced->count(operand1);
  bool replace_op2 = needs_replaced->count(operand2);
  if (replace_op1 || replace_op2)
    {
      std::shared_ptr<MultiplyOperator> new_op = std::make_shared<MultiplyOperator>();
      if (replace_op1)
	{
	  new_op->operand1 = needs_replaced->at(operand1);
	}
      else
	{
	  new_op->operand1 = operand1;
	}
      if (replace_op2)
	{
	  new_op->operand2 = needs_replaced->at(operand2);
	}
      else
	{
	  new_op->operand2 = operand2;
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> AddOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op1 = needs_replaced->count(operand1);
  bool replace_op2 = needs_replaced->count(operand2);
  if (replace_op1 || replace_op2)
    {
      std::shared_ptr<AddOperator> new_op = std::make_shared<AddOperator>();
      if (replace_op1)
	{
	  new_op->operand1 = needs_replaced->at(operand1);
	}
      else
	{
	  new_op->operand1 = operand1;
	}
      if (replace_op2)
	{
	  new_op->operand2 = needs_replaced->at(operand2);
	}
      else
	{
	  new_op->operand2 = operand2;
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> SubtractOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op1 = needs_replaced->count(operand1);
  bool replace_op2 = needs_replaced->count(operand2);
  if (replace_op1 || replace_op2)
    {
      std::shared_ptr<SubtractOperator> new_op = std::make_shared<SubtractOperator>();
      if (replace_op1)
	{
	  new_op->operand1 = needs_replaced->at(operand1);
	}
      else
	{
	  new_op->operand1 = operand1;
	}
      if (replace_op2)
	{
	  new_op->operand2 = needs_replaced->at(operand2);
	}
      else
	{
	  new_op->operand2 = operand2;
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> DivideOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op1 = needs_replaced->count(operand1);
  bool replace_op2 = needs_replaced->count(operand2);
  if (replace_op1 || replace_op2)
    {
      std::shared_ptr<DivideOperator> new_op = std::make_shared<DivideOperator>();
      if (replace_op1)
	{
	  new_op->operand1 = needs_replaced->at(operand1);
	}
      else
	{
	  new_op->operand1 = operand1;
	}
      if (replace_op2)
	{
	  new_op->operand2 = needs_replaced->at(operand2);
	}
      else
	{
	  new_op->operand2 = operand2;
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> PowerOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op1 = needs_replaced->count(operand1);
  bool replace_op2 = needs_replaced->count(operand2);
  if (replace_op1 || replace_op2)
    {
      std::shared_ptr<PowerOperator> new_op = std::make_shared<PowerOperator>();
      if (replace_op1)
	{
	  new_op->operand1 = needs_replaced->at(operand1);
	}
      else
	{
	  new_op->operand1 = operand1;
	}
      if (replace_op2)
	{
	  new_op->operand2 = needs_replaced->at(operand2);
	}
      else
	{
	  new_op->operand2 = operand2;
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> NegationOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op = needs_replaced->count(operand);
  if (replace_op)
    {
      std::shared_ptr<NegationOperator> new_op = std::make_shared<NegationOperator>();
      new_op->operand = needs_replaced->at(operand);
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> ExpOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op = needs_replaced->count(operand);
  if (replace_op)
    {
      std::shared_ptr<ExpOperator> new_op = std::make_shared<ExpOperator>();
      new_op->operand = needs_replaced->at(operand);
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> LogOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_op = needs_replaced->count(operand);
  if (replace_op)
    {
      std::shared_ptr<LogOperator> new_op = std::make_shared<LogOperator>();
      new_op->operand = needs_replaced->at(operand);
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<Operator> ExternalOperator::replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  bool replace_any_op = false;
  bool replace_op;

  for (std::shared_ptr<Node> &op : (*operands))
    {
      replace_op = needs_replaced->count(op);
      replace_any_op = replace_any_op || replace_op;
    }

  if (replace_any_op)
    {
      std::shared_ptr<ExternalOperator> new_op = std::make_shared<ExternalOperator>();
      for (std::shared_ptr<Node> &op : (*operands))
	{
	  replace_op = needs_replaced->count(op);
	  if (replace_op)
	    {
	      new_op->operands->push_back(needs_replaced->at(op));
	    }
	  else
	    {
	      new_op->operands->push_back(op);
	    }
	}
      (*needs_replaced)[shared_from_this()] = new_op;
      return new_op;
    }
  else
    {
      return shared_from_this();
    }
}


std::shared_ptr<ExpressionBase> Expression::distribute_products()
{
  std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process = std::make_shared<std::vector<std::shared_ptr<Operator> > >();
  std::shared_ptr<std::vector<std::shared_ptr<Operator> > > new_operators = std::make_shared<std::vector<std::shared_ptr<Operator> > >();
  std::shared_ptr<std::unordered_set<std::shared_ptr<Node> > > already_processed = std::make_shared<std::unordered_set<std::shared_ptr<Node> > >();
  std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced = std::make_shared<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > >();

  for (std::shared_ptr<Operator> &op : (*operators))
    {
      operators_to_process->push_back(op);
    }

  std::shared_ptr<Operator> op;
  while (operators_to_process->size() > 0)
    {
      op = operators_to_process->back();
      operators_to_process->pop_back();
      if (already_processed->count(op))
	{
	  continue;
	}
      op->distribute_products(new_operators, operators_to_process, already_processed, needs_replaced);
    }

  std::shared_ptr<Expression> res = std::make_shared<Expression>();
  res->n_operators = new_operators->size();
  int ndx = new_operators->size() - 1;

  if (needs_replaced->size() > 0)
    {
      while (ndx >= 0)
	{
	  op = new_operators->at(ndx);
	  op = op->replace_operands(needs_replaced);
	  res->operators->push_back(op);
	  ndx -= 1;
	}      
    }
  else
    {
      while (ndx >= 0)
	{
	  op = new_operators->at(ndx);
	  res->operators->push_back(op);
	  ndx -= 1;
	}
    }

  return res;
}


std::shared_ptr<ExpressionBase> Leaf::distribute_products()
{
  return shared_from_this();
}


void Operator::distribute_products(std::shared_ptr<std::vector<std::shared_ptr<Operator> > > new_operators, std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process, std::shared_ptr<std::unordered_set<std::shared_ptr<Node> > > already_processed, std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  new_operators->push_back(shared_from_this());
}


void _add_product(std::shared_ptr<Node> op1, std::shared_ptr<Node> op2, std::shared_ptr<SumOperator> _sum, std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process)
{
  std::shared_ptr<MultiplyOperator> m = std::make_shared<MultiplyOperator>();
  m->operand1 = op1;
  m->operand2 = op2;
  _sum->operands->push_back(m);
  operators_to_process->push_back(m);
}


void _subtract_product(std::shared_ptr<Node> op1, std::shared_ptr<Node> op2, std::shared_ptr<SumOperator> _sum, std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process)
{
  std::shared_ptr<MultiplyOperator> m = std::make_shared<MultiplyOperator>();
  m->operand1 = op1;
  m->operand2 = op2;
  std::shared_ptr<NegationOperator> _neg = std::make_shared<NegationOperator>();
  _neg->operand = m;
  _sum->operands->push_back(_neg);
  operators_to_process->push_back(m);
  operators_to_process->push_back(_neg);
}


void MultiplyOperator::distribute_products(std::shared_ptr<std::vector<std::shared_ptr<Operator> > > new_operators, std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process, std::shared_ptr<std::unordered_set<std::shared_ptr<Node> > > already_processed, std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced)
{
  if (operand1->is_add_operator())
    {
      std::shared_ptr<AddOperator> op1 = std::dynamic_pointer_cast<AddOperator>(operand1);
      std::shared_ptr<SumOperator> _sum = std::make_shared<SumOperator>();
      if (operand2->is_add_operator())
	{
	  std::shared_ptr<AddOperator> op2 = std::dynamic_pointer_cast<AddOperator>(operand2);
	  _add_product(op1->operand1, op2->operand1, _sum, operators_to_process);
	  _add_product(op1->operand1, op2->operand2, _sum, operators_to_process);
	  _add_product(op1->operand2, op2->operand1, _sum, operators_to_process);
	  _add_product(op1->operand2, op2->operand2, _sum, operators_to_process);
	}
      else if (operand2->is_subtract_operator())
	{
	  std::shared_ptr<SubtractOperator> op2 = std::dynamic_pointer_cast<SubtractOperator>(operand2);
	  _add_product(op1->operand1, op2->operand1, _sum, operators_to_process);
	  _subtract_product(op1->operand1, op2->operand2, _sum, operators_to_process);
	  _add_product(op1->operand2, op2->operand1, _sum, operators_to_process);
	  _subtract_product(op1->operand2, op2->operand2, _sum, operators_to_process);
	}
      else if (operand2->is_sum_operator())
	{
	  std::shared_ptr<SumOperator> op2 = std::dynamic_pointer_cast<SumOperator>(operand2);
	  for (std::shared_ptr<Node> op2_op : *(op2->operands))
	    {
	      _add_product(op1->operand1, op2_op, _sum, operators_to_process);
	      _add_product(op1->operand2, op2_op, _sum, operators_to_process);
	    }
	}
      else
	{
	  _add_product(op1->operand1, operand2, _sum, operators_to_process);
	  _add_product(op1->operand2, operand2, _sum, operators_to_process);
	}
      new_operators->push_back(_sum);
      already_processed->insert(operand1);
      already_processed->insert(operand2);
      (*needs_replaced)[shared_from_this()] = _sum;
    }
  else if (operand1->is_subtract_operator())
    {
      std::shared_ptr<SubtractOperator> op1 = std::dynamic_pointer_cast<SubtractOperator>(operand1);
      std::shared_ptr<SumOperator> _sum = std::make_shared<SumOperator>();
      if (operand2->is_add_operator())
	{
	  std::shared_ptr<AddOperator> op2 = std::dynamic_pointer_cast<AddOperator>(operand2);
	  _add_product(op1->operand1, op2->operand1, _sum, operators_to_process);
	  _add_product(op1->operand1, op2->operand2, _sum, operators_to_process);
	  _subtract_product(op1->operand2, op2->operand1, _sum, operators_to_process);
	  _subtract_product(op1->operand2, op2->operand2, _sum, operators_to_process);
	}
      else if (operand2->is_subtract_operator())
	{
	  std::shared_ptr<SubtractOperator> op2 = std::dynamic_pointer_cast<SubtractOperator>(operand2);
	  _add_product(op1->operand1, op2->operand1, _sum, operators_to_process);
	  _subtract_product(op1->operand1, op2->operand2, _sum, operators_to_process);
	  _subtract_product(op1->operand2, op2->operand1, _sum, operators_to_process);
	  _add_product(op1->operand2, op2->operand2, _sum, operators_to_process);
	}
      else if (operand2->is_sum_operator())
	{
	  std::shared_ptr<SumOperator> op2 = std::dynamic_pointer_cast<SumOperator>(operand2);
	  for (std::shared_ptr<Node> op2_op : *(op2->operands))
	    {
	      _add_product(op1->operand1, op2_op, _sum, operators_to_process);
	      _subtract_product(op1->operand2, op2_op, _sum, operators_to_process);
	    }
	}
      else
	{
	  _add_product(op1->operand1, operand2, _sum, operators_to_process);
	  _subtract_product(op1->operand2, operand2, _sum, operators_to_process);
	}
      new_operators->push_back(_sum);
      already_processed->insert(operand1);
      already_processed->insert(operand2);
      (*needs_replaced)[shared_from_this()] = _sum;
    }
  else if (operand1->is_sum_operator())
    {
      std::shared_ptr<SumOperator> op1 = std::dynamic_pointer_cast<SumOperator>(operand1);
      std::shared_ptr<SumOperator> _sum = std::make_shared<SumOperator>();
      for (std::shared_ptr<Node> op1_op : *(op1->operands))
	{
	  if (operand2->is_add_operator())
	    {
	      std::shared_ptr<AddOperator> op2 = std::dynamic_pointer_cast<AddOperator>(operand2);
	      _add_product(op1_op, op2->operand1, _sum, operators_to_process);
	      _add_product(op1_op, op2->operand2, _sum, operators_to_process);
	    }
	  else if (operand2->is_subtract_operator())
	    {
	      std::shared_ptr<SubtractOperator> op2 = std::dynamic_pointer_cast<SubtractOperator>(operand2);
	      _add_product(op1_op, op2->operand1, _sum, operators_to_process);
	      _subtract_product(op1_op, op2->operand2, _sum, operators_to_process);
	    }
	  else if (operand2->is_sum_operator())
	    {
	      std::shared_ptr<SumOperator> op2 = std::dynamic_pointer_cast<SumOperator>(operand2);
	      for (std::shared_ptr<Node> op2_op : *(op2->operands))
		{
		  _add_product(op1_op, op2_op, _sum, operators_to_process);
		}
	    }
	  else
	    {
	      _add_product(op1_op, operand2, _sum, operators_to_process);
	    }
	}
      new_operators->push_back(_sum);
      already_processed->insert(operand1);
      already_processed->insert(operand2);
      (*needs_replaced)[shared_from_this()] = _sum;
    }
  else
    {
      if (operand2->is_add_operator() || operand2->is_subtract_operator() || operand2->is_sum_operator())
	{
	  std::shared_ptr<SumOperator> _sum = std::make_shared<SumOperator>();
	  if (operand2->is_add_operator())
	    {
	      std::shared_ptr<AddOperator> op2 = std::dynamic_pointer_cast<AddOperator>(operand2);
	      _add_product(operand1, op2->operand1, _sum, operators_to_process);
	      _add_product(operand1, op2->operand2, _sum, operators_to_process);
	    }
	  else if (operand2->is_subtract_operator())
	    {
	      std::shared_ptr<SubtractOperator> op2 = std::dynamic_pointer_cast<SubtractOperator>(operand2);
	      _add_product(operand1, op2->operand1, _sum, operators_to_process);
	      _subtract_product(operand1, op2->operand2, _sum, operators_to_process);
	    }
	  else 
	    {
	      assert (operand2->is_sum_operator());
	      std::shared_ptr<SumOperator> op2 = std::dynamic_pointer_cast<SumOperator>(operand2);
	      for (std::shared_ptr<Node> op2_op : *(op2->operands))
		{
		  _add_product(operand1, op2_op, _sum, operators_to_process);
		}
	    }
	  new_operators->push_back(_sum);
	  already_processed->insert(operand1);
	  already_processed->insert(operand2);
	  (*needs_replaced)[shared_from_this()] = _sum;
	}
      else
	{
	  new_operators->push_back(shared_from_this());
	}
    }
}


bool BinaryOperator::is_binary_operator()
{
  return true;
}


bool UnaryOperator::is_unary_operator()
{
  return true;
}


bool LinearOperator::is_linear_operator()
{
  return true;
}


bool SumOperator::is_sum_operator()
{
  return true;
}


bool MultiplyOperator::is_multiply_operator()
{
  return true;
}


bool AddOperator::is_add_operator()
{
  return true;
}


bool SubtractOperator::is_subtract_operator()
{
  return true;
}


bool DivideOperator::is_divide_operator()
{
  return true;
}


bool PowerOperator::is_power_operator()
{
  return true;
}


bool NegationOperator::is_negation_operator()
{
  return true;
}


bool ExpOperator::is_exp_operator()
{
  return true;
}


bool LogOperator::is_log_operator()
{
  return true;
}


bool ExternalOperator::is_external_operator()
{
  return true;
}


std::string Repn::__str__()
{
  std::string res = "constant: ";
  res += constant->__str__();
  res += "\n";
  res += "linear: ";
  res += linear->__str__();
  res += "\n";
  res += "quadratic: ";
  res += quadratic->__str__();
  res += "\n";
  res += "nonlinear: ";
  res += nonlinear->__str__();
  return res;
}


std::vector<std::shared_ptr<Repn> > generate_repns(std::vector<std::shared_ptr<ExpressionBase> > exprs)
{
  std::vector<std::shared_ptr<Repn> > res;
  for (std::shared_ptr<ExpressionBase> &e : exprs)
    {
      res.push_back(e->distribute_products()->generate_repn());
      //res.push_back(e->generate_repn());
    }
  return res;
}


std::vector<std::shared_ptr<Var> > create_vars(int n_vars)
{
  std::vector<std::shared_ptr<Var> > res;
  for (int i=0; i<n_vars; ++i)
    {
      res.push_back(std::make_shared<Var>());
    }
  return res;
}


std::vector<std::shared_ptr<Param> > create_params(int n_params)
{
  std::vector<std::shared_ptr<Param> > res;
  for (int i=0; i<n_params; ++i)
    {
      res.push_back(std::make_shared<Param>());
    }
  return res;
}


std::vector<std::shared_ptr<Constant> > create_constants(int n_constants)
{
  std::vector<std::shared_ptr<Constant> > res;
  for (int i=0; i<n_constants; ++i)
    {
      res.push_back(std::make_shared<Constant>());
    }
  return res;
}


std::vector<py::object> generate_prefix_notation(py::object expr)
{
  py::int_ ione = 1;
  py::float_ fone = 1.0;
  py::type int_ = py::type::of(ione);
  py::type float_ = py::type::of(fone);
  py::bool_ is_expression_type;

  std::vector<py::object> res;

  py::type tmp_type = py::type::of(expr);

  if (tmp_type.is(int_) || tmp_type.is(float_))
    {
      res.push_back(expr);
      return res;
    }

  is_expression_type = expr.attr("is_expression_type")();
  if (!is_expression_type)
    {
      res.push_back(expr);
      return res;
    }

  std::vector<py::object> expr_stack;
  expr_stack.push_back(expr);
  py::list reversed_args;
  py::object builtins = py::module_::import("builtins");
  py::object reversed = builtins.attr("reversed");
  
  while (expr_stack.size() > 0)
    {
      expr = expr_stack.back();
      res.push_back(expr);
      expr_stack.pop_back();

      tmp_type = py::type::of(expr);

      if (tmp_type.is(int_) || tmp_type.is(float_))
	{
	  continue;
	}

      is_expression_type = expr.attr("is_expression_type")();
      if (!is_expression_type)
	{
	  continue;
	}

      reversed_args = py::list(reversed(expr.attr("args")));
      for (py::handle arg : reversed_args)
	{
	  expr_stack.push_back(arg.cast<py::object>());
	}
    }

  return res;
}


/*
int main()
{
  std::shared_ptr<Param> a = std::make_shared<Param>();
  a->value = 2.0;
  std::shared_ptr<Var> x = std::make_shared<Var>();
  x->value = 3.0;
  std::shared_ptr<ExpressionBase> expr;
  double val;
  std::shared_ptr<Repn> repn;
  clock_t t0 = clock();
  expr = (*a)*(*x);
  for (int i=0; i<999; ++i)
    {
      expr = *expr + *((*a)*(*x));
    }
  clock_t t1 = clock();
  for (int i=0; i<1000; ++i)
    {
      val = expr->evaluate();
    }
  clock_t t2 = clock();
  std::shared_ptr<Expression> expr2 = std::dynamic_pointer_cast<Expression>(expr);
  for (int i=0; i<1000; ++i)
    {
      repn = expr2->generate_repn();
    }
  clock_t t3 = clock();
  std::cout << ((float)(t1-t0))/CLOCKS_PER_SEC << std::endl;
  std::cout << ((float)(t2-t1))/CLOCKS_PER_SEC << std::endl;
  std::cout << ((float)(t3-t2))/CLOCKS_PER_SEC << std::endl;
  return 0;
  }
*/
