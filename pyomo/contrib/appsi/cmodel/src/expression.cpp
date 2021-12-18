#include "expression.hpp"


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


double Var::get_lb()
{
  double lb1 = lb->evaluate();
  double lb2;
  if (domain == "reals")
    {
      lb2 = -inf;
    }
  else if (domain == "nonnegative_reals")
    {
      lb2 = 0;
    }
  else if (domain == "nonpositive_reals")
    {
      lb2 = -inf;
    }
  else if (domain == "integers")
    {
      lb2 = -inf;
    }
  else if (domain == "nonnegative_integers")
    {
      lb2 = 0;
    }
  else if (domain == "nonpositive_integers")
    {
      lb2 = -inf;
    }
  else if (domain == "binary")
    {
      lb2 = 0;
    }
  else if (domain == "percent_fraction")
    {
      lb2 = 0;
    }
  else if (domain == "unit_interval")
    {
      lb2 = 0;
    }
  else
    {
      throw py::value_error("Unrecognized domain: " + domain);
    }
  return std::max(lb1, lb2);
}


double Var::get_ub()
{
  double ub1 = ub->evaluate();
  double ub2;
  if (domain == "reals")
    {
      ub2 = inf;
    }
  else if (domain == "nonnegative_reals")
    {
      ub2 = inf;
    }
  else if (domain == "nonpositive_reals")
    {
      ub2 = 0;
    }
  else if (domain == "integers")
    {
      ub2 = inf;
    }
  else if (domain == "nonnegative_integers")
    {
      ub2 = inf;
    }
  else if (domain == "nonpositive_integers")
    {
      ub2 = 0;
    }
  else if (domain == "binary")
    {
      ub2 = 1;
    }
  else if (domain == "percent_fraction")
    {
      ub2 = 1;
    }
  else if (domain == "unit_interval")
    {
      ub2 = 1;
    }
  else
    {
      throw py::value_error("Unrecognized domain: " + domain);
    }
  return std::min(ub1, ub2);
}


std::string Var::get_domain()
{
  std::string res;
  if (domain == "reals")
    {
      res = "continuous";
    }
  else if (domain == "nonnegative_reals")
    {
      res = "continuous";
    }
  else if (domain == "nonpositive_reals")
    {
      res = "continuous";
    }
  else if (domain == "integers")
    {
      res = "integers";
    }
  else if (domain == "nonnegative_integers")
    {
      res = "integers";
    }
  else if (domain == "nonpositive_integers")
    {
      res = "integers";
    }
  else if (domain == "binary")
    {
      res = "binary";
    }
  else if (domain == "percent_fraction")
    {
      res = "continuous";
    }
  else if (domain == "unit_interval")
    {
      res = "continuous";
    }
  else
    {
      throw py::value_error("Unrecognized domain: " + domain);
    }
  return res;
}


bool Operator::is_operator_type()
{
  return true;
}


std::vector<std::shared_ptr<Operator> > Expression::get_operators()
{
  std::vector<std::shared_ptr<Operator> > res (n_operators);
  for (unsigned int i=0; i<n_operators; ++i)
    {
      res[i] = operators[i];
    }
  return res;
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
  throw std::runtime_error("cannot evaluate ExternalOperator yet");
}


void LinearOperator::evaluate(double* values)
{
  values[index] = constant->evaluate();
  for (unsigned int i=0; i<nterms; ++i)
    {
      values[index] += coefficients[i]->evaluate() * variables[i]->evaluate();
    }
}


void SumOperator::evaluate(double* values)
{
  values[index] = 0.0;
  for (unsigned int i=0; i<nargs; ++i)
    {
      values[index] += operands[i]->get_value_from_array(values);
    }
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


void Log10Operator::evaluate(double* values)
{
  values[index] = std::log10(operand->get_value_from_array(values));
}


void SinOperator::evaluate(double* values)
{
  values[index] = std::sin(operand->get_value_from_array(values));
}


void CosOperator::evaluate(double* values)
{
  values[index] = std::cos(operand->get_value_from_array(values));
}


void TanOperator::evaluate(double* values)
{
  values[index] = std::tan(operand->get_value_from_array(values));
}


void AsinOperator::evaluate(double* values)
{
  values[index] = std::asin(operand->get_value_from_array(values));
}


void AcosOperator::evaluate(double* values)
{
  values[index] = std::acos(operand->get_value_from_array(values));
}


void AtanOperator::evaluate(double* values)
{
  values[index] = std::atan(operand->get_value_from_array(values));
}


double Expression::evaluate()
{
  double* values = new double[n_operators];
  for (unsigned int i=0; i<n_operators; ++i)
    {
      operators[i]->index = i;
      operators[i]->evaluate(values);
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
  for (unsigned int i=0; i<nargs; ++i)
    {
      if (operands[i]->is_variable_type())
	{
	  var_set.insert(operands[i]);
	}
    }
}


void LinearOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  for (unsigned int i=0; i<nterms; ++i)
    {
      var_set.insert(variables[i]);
    }
}


void SumOperator::identify_variables(std::set<std::shared_ptr<Node> > &var_set)
{
  for (unsigned int i=0; i<nargs; ++i)
    {
      if (operands[i]->is_variable_type())
	{
	  var_set.insert(operands[i]);
	}
    }
}


std::shared_ptr<std::vector<std::shared_ptr<Var> > > Expression::identify_variables()
{
  std::set<std::shared_ptr<Node> > var_set;
  for (unsigned int i=0; i<n_operators; ++i)
    {
      operators[i]->identify_variables(var_set);
    }
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > res = std::make_shared<std::vector<std::shared_ptr<Var> > >(var_set.size());
  int ndx = 0;
  for (std::shared_ptr<Node> v : var_set)
    {
      (*res)[ndx] = std::dynamic_pointer_cast<Var>(v);
      ndx += 1;
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
      if (operators[i]->is_external_operator())
	{
	  external_set.insert(operators[i]);
	}
    }
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > res = std::make_shared<std::vector<std::shared_ptr<ExternalOperator> > >(external_set.size());
  int ndx = 0;
  for (std::shared_ptr<Node> n : external_set)
    {
      (*res)[ndx] = std::dynamic_pointer_cast<ExternalOperator>(n);
      ndx += 1;
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


void LinearOperator::propagate_degree_forward(int* degrees, double* values)
{
  degrees[index] = 1;
}


void SumOperator::propagate_degree_forward(int* degrees, double* values)
{
  int deg = 0;
  int _deg;
  for (unsigned int i=0; i<nargs; ++i)
    {
      _deg = operands[i]->get_degree_from_array(degrees);
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


void UnaryOperator::propagate_degree_forward(int* degrees, double* values)
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
      oper = operators[i];
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
  for (unsigned int i=0; i<(nargs - 1); ++i)
    {
      res += operands[i]->get_string_from_array(string_array);
      res += ", ";
    }
  res += operands[nargs-1]->get_string_from_array(string_array);
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


void Log10Operator::print(std::string* string_array)
{
  string_array[index] = ("log10(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void SinOperator::print(std::string* string_array)
{
  string_array[index] = ("sin(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void CosOperator::print(std::string* string_array)
{
  string_array[index] = ("cos(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void TanOperator::print(std::string* string_array)
{
  string_array[index] = ("tan(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void AsinOperator::print(std::string* string_array)
{
  string_array[index] = ("asin(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void AcosOperator::print(std::string* string_array)
{
  string_array[index] = ("acos(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void AtanOperator::print(std::string* string_array)
{
  string_array[index] = ("atan(" +
			 operand->get_string_from_array(string_array) +
			 ")");
}


void LinearOperator::print(std::string* string_array)
{
  std::string res = "(" + constant->__str__();
  for (unsigned int i=0; i<nterms; ++i)
    {
      res += " + " + coefficients[i]->__str__() + "*" + variables[i]->__str__();
    }
  res += ")";
  string_array[index] = res;
}


void SumOperator::print(std::string* string_array)
{
  std::string res = "(" + operands[0]->get_string_from_array(string_array);
  for (unsigned int i=1; i<nargs; ++ i)
    {
      res += " + " + operands[i]->get_string_from_array(string_array);
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
  stack->push_back(operators[n_operators-1]);
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
  int ndx = nargs - 1;
  while (ndx >= 0)
    {
      stack->push_back(operands[ndx]);
      ndx -= 1;
    }
}


void LinearOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  ; // This is treated as a leaf in this context; write_nl_string will take care of it
}


void ExternalOperator::fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack)
{
  int i = nargs - 1;
  while (i >= 0)
    {
      stack->push_back(operands[i]);
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
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > prefix_notation = get_prefix_notation();
  for (std::shared_ptr<Node> &node : *(prefix_notation))
    {
      node->write_nl_string(f);
    }
}


void MultiplyOperator::write_nl_string(std::ofstream& f)
{
  f << "o2\n";
}


void ExternalOperator::write_nl_string(std::ofstream& f)
{
  f << "f" << external_function_index << " " << nargs << "\n";
}


void SumOperator::write_nl_string(std::ofstream& f)
{
  if (nargs == 2)
    {
      f << "o0\n";
    }
  else
    {
      f << "o54\n";
      f << nargs << "\n";
    }
}


void LinearOperator::write_nl_string(std::ofstream& f)
{
  bool has_const = (!constant->is_constant_type()) || (constant->evaluate() != 0);
  if (has_const)
    {
      if (nterms == 1)
	{
	  f << "o0\n";
	}
      else
	{
	  f << "o54\n";
	  f << nterms + 1 << "\n";
	}
      f << "n" << constant->evaluate() << "\n";
    }
  else
    {
      if (nterms == 2)
	{
	  f << "o0\n";
	}
      else
	{
	  f << "o54\n";
	  f << nterms << "\n";
	}
    }
  for (unsigned int ndx=0; ndx<nterms; ++ndx)
    {
      f << "o2\n";
      f << "n" <<  coefficients[ndx]->evaluate() << "\n";
      variables[ndx]->write_nl_string(f);
    }
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


void Log10Operator::write_nl_string(std::ofstream& f)
{
  f << "o42\n";
}


void SinOperator::write_nl_string(std::ofstream& f)
{
  f << "o41\n";
}


void CosOperator::write_nl_string(std::ofstream& f)
{
  f << "o46\n";
}


void TanOperator::write_nl_string(std::ofstream& f)
{
  f << "o38\n";
}


void AsinOperator::write_nl_string(std::ofstream& f)
{
  f << "o51\n";
}


void AcosOperator::write_nl_string(std::ofstream& f)
{
  f << "o53\n";
}


void AtanOperator::write_nl_string(std::ofstream& f)
{
  f << "o49\n";
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


void Leaf::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  ;
}


void Expression::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  throw std::runtime_error("This should not happen");
}


void BinaryOperator::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  oper_ndx -= 1;
  oper_array[oper_ndx] = shared_from_this();
  // The order does not actually matter here. It
  // will just be easier to debug this way.
  operand2->fill_expression(oper_array, oper_ndx);
  operand1->fill_expression(oper_array, oper_ndx);
}


void UnaryOperator::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  oper_ndx -= 1;
  oper_array[oper_ndx] = shared_from_this();
  operand->fill_expression(oper_array, oper_ndx);
}


void LinearOperator::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  oper_ndx -= 1;
  oper_array[oper_ndx] = shared_from_this();
}


void SumOperator::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  oper_ndx -= 1;
  oper_array[oper_ndx] = shared_from_this();
  // The order does not actually matter here. It
  // will just be easier to debug this way.
  int arg_ndx = nargs - 1;
  while (arg_ndx >= 0)
    {
      operands[arg_ndx]->fill_expression(oper_array, oper_ndx);
      arg_ndx -= 1;
    }
}


void ExternalOperator::fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx)
{
  oper_ndx -= 1;
  oper_array[oper_ndx] = shared_from_this();
  // The order does not actually matter here. It
  // will just be easier to debug this way.
  int arg_ndx = nargs - 1;
  while (arg_ndx >= 0)
    {
      operands[arg_ndx]->fill_expression(oper_array, oper_ndx);
      arg_ndx -= 1;
    }
}


double Leaf::get_lb_from_array(double* lbs)
{
  return value;
}


double Leaf::get_ub_from_array(double* ubs)
{
  return value;
}


double Var::get_lb_from_array(double* lbs)
{
  return get_lb();
}


double Var::get_ub_from_array(double* ubs)
{
  return get_ub();
}


double Expression::get_lb_from_array(double* lbs)
{
  return lbs[n_operators-1];
}


double Expression::get_ub_from_array(double* ubs)
{
  return ubs[n_operators-1];
}


double Operator::get_lb_from_array(double* lbs)
{
  return lbs[index];
}


double Operator::get_ub_from_array(double* ubs)
{
  return ubs[index];
}


void Leaf::set_bounds_in_array(double new_lb, double new_ub, double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  if (new_lb < value - feasibility_tol || new_lb > value + feasibility_tol)
    {
      throw py::value_error("Infeasible constraint");
    }

  if (new_ub < value - feasibility_tol || new_ub > value + feasibility_tol)
    {
      throw py::value_error("Infeasible constraint");
    }
}


void Var::set_bounds_in_array(double new_lb, double new_ub, double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  double orig_lb = get_lb();
  double orig_ub = get_ub();
  
  if (new_lb > new_ub)
    {
      if (new_lb - feasibility_tol > new_ub)
	{
	  throw py::value_error("Infeasible constraint");
	}
      else
	{
	  new_lb -= feasibility_tol;
	  new_ub += feasibility_tol;
	}
    }
  if (new_lb >= inf)
    {
      throw py::value_error("Infeasible constraint");
    }
  if (new_ub <= -inf)
    {
      throw py::value_error("Infeasible constraint");
    }

  if (get_domain() == "integers" || get_domain() == "binary")
    {
      if (new_lb > -inf)
	{
	  double lb_floor = floor(new_lb);
	  double lb_ceil = ceil(new_lb - integer_tol);
	  if (lb_floor > lb_ceil)
	    {
	      new_lb = lb_floor;
	    }
	  else
	    {
	      new_lb = lb_ceil;
	    }
	}
      if (new_ub < inf)
	{
	  double ub_ceil = ceil(new_ub);
	  double ub_floor = floor(new_ub + integer_tol);
	  if (ub_ceil < ub_floor)
	    {
	      new_ub = ub_ceil;
	    }
	  else
	    {
	      new_ub = ub_floor;
	    }
	}
    }

  if (new_lb > orig_lb)
    {
      if (lb->is_leaf())
	{
	  std::dynamic_pointer_cast<Leaf>(lb)->value = new_lb;
	}
      else
	{
	  throw py::value_error("variable bounds cannot be expressions when performing FBBT");
	}
    }

  if (new_ub < orig_ub)
    {
      if (ub->is_leaf())
	{
	  std::dynamic_pointer_cast<Leaf>(ub)->value = new_ub;
	}
      else
	{
	  throw py::value_error("variable bounds cannot be expressions when performing FBBT");
	}
    }
}


void Expression::set_bounds_in_array(double new_lb, double new_ub, double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  lbs[n_operators - 1] = new_lb;
  ubs[n_operators - 1] = new_ub;
}


void Operator::set_bounds_in_array(double new_lb, double new_ub, double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  lbs[index] = new_lb;
  ubs[index] = new_ub;
}


void Expression::propagate_bounds_forward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  for (unsigned int ndx=0; ndx < n_operators; ++ndx)
    {
      operators[ndx]->index = ndx;
      operators[ndx]->propagate_bounds_forward(lbs, ubs, feasibility_tol, integer_tol);
    }
}


void Expression::propagate_bounds_backward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  int ndx = n_operators - 1;
  while (ndx >= 0)
    {
      operators[ndx]->propagate_bounds_backward(lbs, ubs, feasibility_tol, integer_tol);
      ndx -= 1;
    }
}


void Operator::propagate_bounds_forward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  lbs[index] = -inf;
  ubs[index] = inf;
}


void Operator::propagate_bounds_backward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  ;
}


void MultiplyOperator::propagate_bounds_forward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  if (operand1 == operand2)
    {
      throw py::value_error("this is not implemented yet");
    }
  else
    {
      mul(operand1->get_lb_from_array(lbs),
	  operand1->get_ub_from_array(ubs),
	  operand2->get_lb_from_array(lbs),
	  operand2->get_ub_from_array(ubs),
	  &lbs[index],
	  &ubs[index]);
    }
}


void MultiplyOperator::propagate_bounds_backward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  double xl = operand1->get_lb_from_array(lbs);
  double xu = operand1->get_ub_from_array(ubs);
  double yl = operand2->get_lb_from_array(lbs);
  double yu = operand2->get_ub_from_array(ubs);
  double lb = get_lb_from_array(lbs);
  double ub = get_ub_from_array(ubs);

  double new_xl;
  double new_xu;
  double new_yl;
  double new_yu;

  if (operand1 == operand2)
    {
      throw py::value_error("this is not implemented yet");
    }
  else
    {
      div(lb, ub, yl, yu, &new_xl, &new_xu, feasibility_tol);
      div(lb, ub, xl, xu, &new_yl, &new_yu, feasibility_tol);
    }

  if (new_xl > xl)
    {
      xl = new_xl;
    }
  if (new_xu < xu)
    {
      xu = new_xu;
    }
  operand1->set_bounds_in_array(xl, xu, lbs, ubs, feasibility_tol, integer_tol);

  if (new_yl > yl)
    {
      yl = new_yl;
    }
  if (new_yu < yu)
    {
      yu = new_yu;
    }
  operand2->set_bounds_in_array(yl, yu, lbs, ubs, feasibility_tol, integer_tol);
}


void SumOperator::propagate_bounds_forward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  double lb = operands[0]->get_lb_from_array(lbs);
  double ub = operands[0]->get_ub_from_array(ubs);
  double tmp_lb;
  double tmp_ub;

  for (unsigned int ndx=1; ndx < nargs; ++ndx)
    {
      add(lb, ub, operands[ndx]->get_lb_from_array(lbs), operands[ndx]->get_ub_from_array(ubs), &tmp_lb, &tmp_ub);
      lb = tmp_lb;
      ub = tmp_ub;
    }

  lbs[index] = lb;
  ubs[index] = ub;
}


void SumOperator::propagate_bounds_backward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  double* accumulated_lbs = new double[nargs];
  double* accumulated_ubs = new double[nargs];

  accumulated_lbs[0] = operands[0]->get_lb_from_array(lbs);
  accumulated_ubs[0] = operands[0]->get_ub_from_array(ubs);
  for (unsigned int ndx=1; ndx < nargs; ++ndx)
    {
      add(accumulated_lbs[ndx-1],
	  accumulated_ubs[ndx-1],
	  operands[ndx]->get_lb_from_array(lbs),
	  operands[ndx]->get_ub_from_array(ubs),
	  &accumulated_lbs[ndx],
	  &accumulated_ubs[ndx]);
    }

  double new_sum_lb = get_lb_from_array(lbs);
  double new_sum_ub = get_ub_from_array(ubs);

  if (new_sum_lb > accumulated_lbs[nargs-1])
    {
      accumulated_lbs[nargs-1] = new_sum_lb;
    }
  if (new_sum_ub < accumulated_ubs[nargs-1])
    {
      accumulated_ubs[nargs-1] = new_sum_ub;
    }

  double lb0;
  double ub0;
  double lb1;
  double ub1;
  double lb2;
  double ub2;
  double _lb1;
  double _ub1;
  double _lb2;
  double _ub2;

  int ndx = nargs - 1;
  while (ndx >= 1)
    {
      lb0 = accumulated_lbs[ndx];
      ub0 = accumulated_ubs[ndx];
      lb1 = accumulated_lbs[ndx - 1];
      ub1 = accumulated_ubs[ndx - 1];
      lb2 = operands[ndx]->get_lb_from_array(lbs);
      ub2 = operands[ndx]->get_ub_from_array(ubs);
      sub(lb0, ub0, lb2, ub2, &_lb1, &_ub1);
      sub(lb0, ub0, lb1, ub1, &_lb2, &_ub2);
      if (_lb1 > lb1)
	{
	  lb1 = _lb1;
	}
      if (_ub1 < ub1)
	{
	  ub1 = _ub1;
	}
      if (_lb2 > lb2)
	{
	  lb2 = _lb2;
	}
      if (_ub2 < ub2)
	{
	  ub2 = _ub2;
	}
      accumulated_lbs[ndx-1] = lb1;
      accumulated_ubs[ndx-1] = ub1;
      operands[ndx]->set_bounds_in_array(lb2, ub2, lbs, ubs, feasibility_tol, integer_tol);
      ndx -= 1;
    }

  // take care of ndx = 0
  lb1 = operands[0]->get_lb_from_array(lbs);
  ub1 = operands[0]->get_ub_from_array(ubs);
  _lb1 = accumulated_lbs[0];
  _ub1 = accumulated_ubs[0];
  if (_lb1 > lb1)
    {
      lb1 = _lb1;
    }
  if (_ub1 < ub1)
    {
      ub1 = _ub1;
    }
  operands[0]->set_bounds_in_array(lb1, ub1, lbs, ubs, feasibility_tol, integer_tol);

  delete[] accumulated_lbs;
  delete[] accumulated_ubs;
}


void DivideOperator::propagate_bounds_forward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  div(operand1->get_lb_from_array(lbs),
      operand1->get_ub_from_array(ubs),
      operand2->get_lb_from_array(lbs),
      operand2->get_ub_from_array(ubs),
      &lbs[index],
      &ubs[index],
      feasibility_tol);  
}


void DivideOperator::propagate_bounds_backward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  double xl = operand1->get_lb_from_array(lbs);
  double xu = operand1->get_ub_from_array(ubs);
  double yl = operand2->get_lb_from_array(lbs);
  double yu = operand2->get_ub_from_array(ubs);
  double lb = get_lb_from_array(lbs);
  double ub = get_ub_from_array(ubs);

  double new_xl;
  double new_xu;
  double new_yl;
  double new_yu;

  mul(lb, ub, yl, yu, &new_xl, &new_xu);
  div(xl, xu, lb, ub, &new_yl, &new_yu, feasibility_tol);

  if (new_xl > xl)
    {
      xl = new_xl;
    }
  if (new_xu < xu)
    {
      xu = new_xu;
    }
  operand1->set_bounds_in_array(xl, xu, lbs, ubs, feasibility_tol, integer_tol);

  if (new_yl > yl)
    {
      yl = new_yl;
    }
  if (new_yu < yu)
    {
      yu = new_yu;
    }
  operand2->set_bounds_in_array(yl, yu, lbs, ubs, feasibility_tol, integer_tol);
}

void NegationOperator::propagate_bounds_forward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  sub(0, 0, operand->get_lb_from_array(lbs),
      operand->get_ub_from_array(ubs),
      &lbs[index], &ubs[index]);  
}


void NegationOperator::propagate_bounds_backward(double* lbs, double* ubs, double feasibility_tol, double integer_tol)
{
  double xl = operand->get_lb_from_array(lbs);
  double xu = operand->get_ub_from_array(ubs);
  double lb = get_lb_from_array(lbs);
  double ub = get_ub_from_array(ubs);

  double new_xl;
  double new_xu;

  sub(0, 0, lb, ub, &new_xl, &new_xu);

  if (new_xl > xl)
    {
      xl = new_xl;
    }
  if (new_xu < xu)
    {
      xu = new_xu;
    }
  operand->set_bounds_in_array(xl, xu, lbs, ubs, feasibility_tol, integer_tol);
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


std::shared_ptr<Node> appsi_operator_from_pyomo_expr(py::handle expr, py::handle var_map, py::handle param_map, PyomoExprTypes& expr_types)
{
  std::shared_ptr<Node> res;
  int tmp_type = expr_types.expr_type_map[py::type::of(expr)].cast<int>();
  
  switch (tmp_type)
    {
    case 0 :
      {
	res = std::make_shared<Constant>(expr.cast<double>());
	break;
      }
    case 1 :
      {
        res = var_map[expr_types.id(expr)].cast<std::shared_ptr<Node> >();
	break;
      }
    case 2 :
      {
        res = param_map[expr_types.id(expr)].cast<std::shared_ptr<Node> >();
	break;
      }
    case 3 :
      {
        res = std::make_shared<MultiplyOperator>();
	break;
      }
    case 4 :
      {
        res = std::make_shared<SumOperator>(expr.attr("nargs")().cast<int>());
	break;
      }
    case 5 :
      {
        res = std::make_shared<NegationOperator>();
	break;
      }
    case 6 :
      {
        res = std::make_shared<ExternalOperator>(expr.attr("nargs")().cast<int>());
	std::shared_ptr<ExternalOperator> oper = std::dynamic_pointer_cast<ExternalOperator>(res);
	oper->function_name = expr.attr("_fcn").attr("_function").cast<std::string>();
	break;
      }
    case 7 :
      {
        res = std::make_shared<PowerOperator>();
	break;
      }
    case 8 :
      {
        res = std::make_shared<DivideOperator>();
	break;
      }
    case 9 :
      {
        std::string function_name = expr.attr("getname")().cast<std::string>();
	if (function_name == "exp")
	  {
	    res =  std::make_shared<ExpOperator>();
	  }
	else if (function_name == "log")
	  {
	    res =  std::make_shared<LogOperator>();
	  }
	else if (function_name == "log10")
	  {
	    res =  std::make_shared<Log10Operator>();
	  }
	else if (function_name == "sin")
	  {
	    res =  std::make_shared<SinOperator>();
	  }
	else if (function_name == "cos")
	  {
	    res =  std::make_shared<CosOperator>();
	  }
	else if (function_name == "tan")
	  {
	    res =  std::make_shared<TanOperator>();
	  }
	else if (function_name == "asin")
	  {
	    res =  std::make_shared<AsinOperator>();
	  }
	else if (function_name == "acos")
	  {
	    res =  std::make_shared<AcosOperator>();
	  }
	else if (function_name == "atan")
	  {
	    res =  std::make_shared<AtanOperator>();
	  }
	else
	  {
	    throw py::value_error("Unrecognized expression type");
	  }
	break;
      }
    case 10 :
      {
        res = std::make_shared<LinearOperator>(expr_types.len(expr.attr("linear_vars")).cast<int>());
	break;
      }
    case 11 :
      {
	res = appsi_operator_from_pyomo_expr(expr.attr("expr"), var_map, param_map, expr_types);
	break;
      }
    case 12 :
      {
	res = std::make_shared<Constant>(expr.attr("value").cast<double>());
	break;
      }
    default :
      {
        throw py::value_error("Unrecognized expression type");
	break;
      }
    }
  return res;
}


void prep_for_repn_helper(py::handle expr, py::handle named_exprs, py::handle variables, py::handle fixed_vars, py::handle external_funcs, PyomoExprTypes& expr_types)
{
  int tmp_type = expr_types.expr_type_map[py::type::of(expr)].cast<int>();
  
  switch (tmp_type)
    {
    case 0 :
      {
	break;
      }
    case 1 :
      {
	variables[expr_types.id(expr)] = expr;
	if (expr.attr("fixed").cast<bool>())
	  {
	    fixed_vars[expr_types.id(expr)] = expr;
	  }
	break;
      }
    case 2 :
      {
	break;
      }
    case 3 :
      {
	py::tuple args = expr.attr("_args_");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 4 :
      {
	py::tuple args = expr.attr("args");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 5 :
      {
	py::tuple args = expr.attr("_args_");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 6 :
      {
	external_funcs[expr_types.id(expr)] = expr;
	py::tuple args = expr.attr("args");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 7 :
      {
	py::tuple args = expr.attr("_args_");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 8 :
      {
	py::tuple args = expr.attr("_args_");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 9 :
      {
	py::tuple args = expr.attr("_args_");
	for (py::handle arg : args)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	break;
      }
    case 10 :
      {
	py::list linear_vars = expr.attr("linear_vars");
	py::list linear_coefs = expr.attr("linear_coefs");
	for (py::handle arg : linear_vars)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	for (py::handle arg : linear_coefs)
	  {
	    prep_for_repn_helper(arg, named_exprs, variables, fixed_vars, external_funcs, expr_types);
	  }
	prep_for_repn_helper(expr.attr("constant"), named_exprs, variables, fixed_vars, external_funcs, expr_types);
	break;
      }
    case 11 :
      {
	named_exprs[expr_types.id(expr)] = expr;
	prep_for_repn_helper(expr.attr("expr"), named_exprs, variables, fixed_vars, external_funcs, expr_types);
	break;
      }
    case 12 :
      {
	break;
      }
    default :
      {
        throw py::value_error("Unrecognized expression type");
	break;
      }
    }
}


py::tuple prep_for_repn(py::handle expr, PyomoExprTypes& expr_types)
{
  py::dict named_exprs;
  py::dict variables;
  py::dict fixed_vars;
  py::dict external_funcs;

  prep_for_repn_helper(expr, named_exprs, variables, fixed_vars, external_funcs, expr_types);

  py::list named_expr_list = named_exprs.attr("values")();
  py::list variable_list = variables.attr("values")();
  py::list fixed_var_list = fixed_vars.attr("values")();
  py::list external_func_list = external_funcs.attr("values")();

  py::tuple res = py::make_tuple(named_expr_list, variable_list, fixed_var_list, external_func_list);
  return res;
}


int build_expression_tree(py::handle pyomo_expr, std::shared_ptr<Node> appsi_expr, py::handle var_map, py::handle param_map, PyomoExprTypes& expr_types)
{
  int num_nodes = 0;

  if (appsi_expr->is_leaf())
    {
      ;
    }
  else if (appsi_expr->is_binary_operator())
    {
      num_nodes += 1;
      std::shared_ptr<BinaryOperator> oper = std::dynamic_pointer_cast<BinaryOperator>(appsi_expr);
      py::list pyomo_args = pyomo_expr.attr("args");
      oper->operand1 = appsi_operator_from_pyomo_expr(pyomo_args[0], var_map, param_map, expr_types);
      oper->operand2 = appsi_operator_from_pyomo_expr(pyomo_args[1], var_map, param_map, expr_types);
      num_nodes += build_expression_tree(pyomo_args[0], oper->operand1, var_map, param_map, expr_types);
      num_nodes += build_expression_tree(pyomo_args[1], oper->operand2, var_map, param_map, expr_types);
    }
  else if (appsi_expr->is_unary_operator())
    {
      num_nodes += 1;
      std::shared_ptr<UnaryOperator> oper = std::dynamic_pointer_cast<UnaryOperator>(appsi_expr);
      py::list pyomo_args = pyomo_expr.attr("args");
      oper->operand = appsi_operator_from_pyomo_expr(pyomo_args[0], var_map, param_map, expr_types);
      num_nodes += build_expression_tree(pyomo_args[0], oper->operand, var_map, param_map, expr_types);
    }
  else if (appsi_expr->is_sum_operator())
    {
      num_nodes += 1;
      std::shared_ptr<SumOperator> oper = std::dynamic_pointer_cast<SumOperator>(appsi_expr);
      py::list pyomo_args = pyomo_expr.attr("args");
      for (unsigned int arg_ndx=0; arg_ndx < oper->nargs; ++arg_ndx)
	{
	  oper->operands[arg_ndx] = appsi_operator_from_pyomo_expr(pyomo_args[arg_ndx], var_map, param_map, expr_types);
	  num_nodes += build_expression_tree(pyomo_args[arg_ndx], oper->operands[arg_ndx], var_map, param_map, expr_types);
	}
    }
  else if (appsi_expr->is_linear_operator())
    {
      num_nodes += 1;
      std::shared_ptr<LinearOperator> oper = std::dynamic_pointer_cast<LinearOperator>(appsi_expr);
      oper->constant = appsi_expr_from_pyomo_expr(pyomo_expr.attr("constant"), var_map, param_map, expr_types);
      py::list pyomo_vars = pyomo_expr.attr("linear_vars");
      py::list pyomo_coefs = pyomo_expr.attr("linear_coefs");
      for (unsigned int arg_ndx=0; arg_ndx < oper->nterms; ++arg_ndx)
	{
	  oper->variables[arg_ndx] = var_map[expr_types.id(pyomo_vars[arg_ndx])].cast<std::shared_ptr<Var> >();
	  oper->coefficients[arg_ndx] = appsi_expr_from_pyomo_expr(pyomo_coefs[arg_ndx], var_map, param_map, expr_types);
	}
    }
  else if (appsi_expr->is_external_operator())
    {
      num_nodes += 1;
      std::shared_ptr<ExternalOperator> oper = std::dynamic_pointer_cast<ExternalOperator>(appsi_expr);
      py::list pyomo_args = pyomo_expr.attr("args");
      for (unsigned int arg_ndx=0; arg_ndx < oper->nargs; ++arg_ndx)
	{
	  oper->operands[arg_ndx] = appsi_operator_from_pyomo_expr(pyomo_args[arg_ndx], var_map, param_map, expr_types);
	  num_nodes += build_expression_tree(pyomo_args[arg_ndx], oper->operands[arg_ndx], var_map, param_map, expr_types);
	}
    }
  else
    {
      throw py::value_error("Unrecognized expression type");
    }
  return num_nodes;
}


std::shared_ptr<ExpressionBase> appsi_expr_from_pyomo_expr(py::handle expr, py::handle var_map, py::handle param_map, PyomoExprTypes& expr_types)
{
  std::shared_ptr<Node> node = appsi_operator_from_pyomo_expr(expr, var_map, param_map, expr_types);
  int num_nodes = build_expression_tree(expr, node, var_map, param_map, expr_types);
  if (num_nodes == 0)
    {
      return std::dynamic_pointer_cast<ExpressionBase>(node);
    }
  else
    {
      std::shared_ptr<Expression> res = std::make_shared<Expression>(num_nodes);
      node->fill_expression(res->operators, num_nodes);
      return res;
    }
}


std::vector<std::shared_ptr<ExpressionBase> > appsi_exprs_from_pyomo_exprs(py::list expr_list, py::dict var_map, py::dict param_map)
{
  PyomoExprTypes expr_types = PyomoExprTypes();
  int num_exprs = expr_types.builtins.attr("len")(expr_list).cast<int>();
  std::vector<std::shared_ptr<ExpressionBase> > res (num_exprs);  

  int ndx = 0;
  for (py::handle expr : expr_list)
    {
      res[ndx] = appsi_expr_from_pyomo_expr(expr, var_map, param_map, expr_types);
      ndx += 1;
    }
  return res;
}


void process_pyomo_vars(PyomoExprTypes& expr_types,
			py::list pyomo_vars,
			py::dict var_map,
			py::dict param_map,
			py::dict var_attrs,
			py::dict rev_var_map,
			py::bool_ _set_name,
			py::handle symbol_map,
			py::handle labeler,
			py::bool_ _update)
{
  py::tuple v_attrs;
  std::shared_ptr<Var> cv;
  py::handle v_lb;
  py::handle v_ub;
  py::handle v_val;
  bool v_fixed;
  py::handle v_domain;
  bool set_name = _set_name.cast<bool>();
  bool update = _update.cast<bool>();
  
  for (py::handle v : pyomo_vars)
    {
      v_attrs = var_attrs[expr_types.id(v)];
      v_lb = v_attrs[1];
      v_ub = v_attrs[2];
      v_fixed = v_attrs[3].cast<bool>();
      v_domain = v_attrs[4];
      v_val = v_attrs[5];

      if (update)
	{
	  cv = var_map[expr_types.id(v)].cast<std::shared_ptr<Var> >();
	}
      else
	{
	  cv = std::make_shared<Var>();
	}

      if (!(v_lb.is(py::none())))
	{
	  cv->lb = appsi_expr_from_pyomo_expr(v_lb, var_map, param_map, expr_types);
	}
      else
	{
	  cv->lb = std::make_shared<Constant>(-inf);
	}
      if (!(v_ub.is(py::none())))
	{
	  cv->ub = appsi_expr_from_pyomo_expr(v_ub, var_map, param_map, expr_types);
	}
      else
	{
	  cv->ub = std::make_shared<Constant>(inf);
	}

      if (!(v_val.is(py::none())))
	{
	  cv->value = v_val.cast<double>();
	}

      if (v_fixed)
	{
	  cv->fixed = true;
	}
      else
	{
	  cv->fixed = false;
	}

      if (set_name && !update)
	{
	  cv->name = symbol_map.attr("getSymbol")(v, labeler).cast<std::string>();
	}

      if (v_domain.is(expr_types.reals))
	{
	  cv->domain = "reals";
	}
      else if (v_domain.is(expr_types.nonnegative_reals))
	{
	  cv->domain = "nonnegative_reals";
	}
      else if (v_domain.is(expr_types.nonpositive_reals))
	{
	  cv->domain = "nonpositive_reals";
	}
      else if (v_domain.is(expr_types.percent_fraction))
	{
	  cv->domain = "percent_fraction";
	}
      else if (v_domain.is(expr_types.unit_interval))
	{
	  cv->domain = "unit_interval";
	}
      else if (v_domain.is(expr_types.integers))
	{
	  cv->domain = "integers";
	}
      else if (v_domain.is(expr_types.nonnegative_integers))
	{
	  cv->domain = "nonnegative_integers";
	}
      else if (v_domain.is(expr_types.nonpositive_integers))
	{
	  cv->domain = "nonpositive_integers";
	}
      else if (v_domain.is(expr_types.binary))
	{
	  cv->domain = "binary";
	}
      else
	{
	  throw py::value_error("Unrecognized domain");
	}

      if (!update)
	{
	  var_map[expr_types.id(v)] = py::cast(cv);
	  rev_var_map[py::cast(cv)] = v;
	}
    }
}
