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
  der = 0.0;
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
