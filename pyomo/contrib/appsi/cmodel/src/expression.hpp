#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <set>
#include <unordered_set>
#include <sstream>
#include <iterator>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <iterator>
#include <typeinfo>
#include <fstream>
#include <algorithm>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace pybind11::literals;

class Node;
class ExpressionBase;
class Leaf;
class Var;
class Constant;
class Param;
class Expression;
class Operator;
class BinaryOperator;
class UnaryOperator;
class LinearOperator;
class SumOperator;
class MultiplyOperator;
class SubtractOperator;
class DivideOperator;
class PowerOperator;
class NegationOperator;
class ExpOperator;
class LogOperator;
class ExternalOperator;
class PyomoExprTypes;


extern double inf;


class Node: public std::enable_shared_from_this<Node>
{
public:
  Node() = default;
  virtual ~Node() = default;
  virtual bool is_variable_type() {return false;}
  virtual bool is_param_type() {return false;}
  virtual bool is_expression_type() {return false;}
  virtual bool is_operator_type() {return false;}
  virtual bool is_constant_type() {return false;}
  virtual bool is_leaf() {return false;}
  virtual bool is_binary_operator() {return false;}
  virtual bool is_unary_operator() {return false;}
  virtual bool is_linear_operator() {return false;}
  virtual bool is_sum_operator() {return false;}
  virtual bool is_multiply_operator() {return false;}
  virtual bool is_subtract_operator() {return false;}
  virtual bool is_divide_operator() {return false;}
  virtual bool is_power_operator() {return false;}
  virtual bool is_negation_operator() {return false;}
  virtual bool is_exp_operator() {return false;}
  virtual bool is_log_operator() {return false;}
  virtual bool is_external_operator() {return false;}
  virtual double get_value_from_array(double*) = 0;
  virtual int get_degree_from_array(int*) = 0;
  virtual std::string get_string_from_array(std::string*) = 0;
  virtual void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) = 0;
  virtual void write_nl_string(std::ofstream&) = 0;
  virtual void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) = 0;
};


class ExpressionBase: public Node
{
public:
  ExpressionBase() = default;
  virtual double evaluate() = 0;
  virtual std::string __str__() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Var> > > identify_variables() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > identify_external_operators() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Node> > > get_prefix_notation() = 0;
  std::shared_ptr<ExpressionBase> shared_from_this() {return std::static_pointer_cast<ExpressionBase>(Node::shared_from_this());}
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override {;}
};


class Leaf: public ExpressionBase
{
public:
  Leaf() = default;
  Leaf(double value) : value(value) {}
  virtual ~Leaf() = default;
  double value = 0.0;
  bool is_leaf() override;
  double evaluate() override;
  double get_value_from_array(double*) override;
  std::string get_string_from_array(std::string*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > get_prefix_notation() override;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class Constant: public Leaf
{
public:
  Constant() = default;
  Constant(double value) : Leaf(value) {}
  bool is_constant_type() override;
  std::string __str__() override;
  int get_degree_from_array(int*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > identify_external_operators() override;
  void write_nl_string(std::ofstream&) override;
};


class Var: public Leaf
{
public:
  Var() = default;
  Var(double val) : Leaf(val) {}
  Var(std::string _name) : name(_name) {}
  Var(std::string _name, double val) : Leaf(val), name(_name) {}
  std::string name = "v";
  std::string __str__() override;
  double lb = -inf;
  double ub = inf;
  int index = -1;
  bool fixed = false;
  std::string domain = "continuous";  // options are continuous, binary, or integer
  bool is_variable_type() override;
  int get_degree_from_array(int*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > identify_external_operators() override;
  void write_nl_string(std::ofstream&) override;
  std::shared_ptr<Var> shared_from_this() {return std::static_pointer_cast<Var>(Node::shared_from_this());}
};


class Param: public Leaf
{
public:
  Param() = default;
  Param(double val) : Leaf(val) {}
  Param(std::string _name) : name(_name) {}
  Param(std::string _name, double val) : Leaf(val), name(_name) {}
  std::string name = "p";
  std::string __str__() override;
  bool is_param_type() override;
  int get_degree_from_array(int*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > identify_external_operators() override;
  void write_nl_string(std::ofstream&) override;
};


class Expression: public ExpressionBase
{
public:
  Expression(int _n_operators) : ExpressionBase()
  {
    operators = new std::shared_ptr<Operator>[_n_operators];
    n_operators = _n_operators;
  }
  ~Expression()
  {
    delete[] operators;
  }
  std::string __str__() override;
  bool is_expression_type() override;
  double evaluate() override;
  double get_value_from_array(double*) override;
  int get_degree_from_array(int*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > identify_external_operators() override;
  std::string get_string_from_array(std::string*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > get_prefix_notation() override;
  void write_nl_string(std::ofstream&) override;
  std::vector<std::shared_ptr<Operator> > get_operators();
  std::shared_ptr<Operator>* operators;
  unsigned int n_operators;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class Operator: public Node
{
public:
  Operator() = default;
  int index = 0;
  virtual void evaluate(double* values) = 0;
  virtual void propagate_degree_forward(int* degrees, double* values) = 0;
  virtual void identify_variables(std::set<std::shared_ptr<Node> >&) = 0;
  std::shared_ptr<Operator> shared_from_this() {return std::static_pointer_cast<Operator>(Node::shared_from_this());}
  bool is_operator_type() override;
  double get_value_from_array(double*) override;
  int get_degree_from_array(int*) override;
  std::string get_string_from_array(std::string*) override;
  virtual void print(std::string*) = 0;
  virtual std::string name() = 0;
};


class BinaryOperator: public Operator
{
public:
  BinaryOperator() = default;
  virtual ~BinaryOperator() = default;
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  std::shared_ptr<Node> operand1;
  std::shared_ptr<Node> operand2;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  bool is_binary_operator() override;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class UnaryOperator: public Operator
{
public:
  UnaryOperator() = default;
  virtual ~UnaryOperator() = default;
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  std::shared_ptr<Node> operand;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  bool is_unary_operator() override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class LinearOperator: public Operator
{
public:
  LinearOperator(int _nterms)
  {
    variables = new std::shared_ptr<Var>[_nterms];
    coefficients = new std::shared_ptr<ExpressionBase>[_nterms];
    nterms = _nterms;
  }
  ~LinearOperator()
  {
    delete[] variables;
    delete[] coefficients;
  }
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  std::shared_ptr<Var>* variables;
  std::shared_ptr<ExpressionBase>* coefficients;
  std::shared_ptr<ExpressionBase> constant = std::make_shared<Constant>(0);
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "LinearOperator";};
  void write_nl_string(std::ofstream&) override;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  bool is_linear_operator() override;
  unsigned int nterms;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class SumOperator: public Operator
{
public:
  SumOperator(int _nargs)
  {
    operands = new std::shared_ptr<Node>[_nargs];
    nargs = _nargs;
  }
  ~SumOperator()
  {
    delete[] operands;
  }
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "SumOperator";};
  void write_nl_string(std::ofstream&) override;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  bool is_sum_operator() override;
  std::shared_ptr<Node>* operands;
  unsigned int nargs;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class MultiplyOperator: public BinaryOperator
{
public:
  MultiplyOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "MultiplyOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_multiply_operator() override;
};


class ExternalOperator: public Operator
{
public:
  ExternalOperator(int _nargs)
  {
    operands = new std::shared_ptr<Node>[_nargs];
    nargs = _nargs;
  }
  ~ExternalOperator()
  {
    delete[] operands;
  }
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "ExternalOperator";};
  void write_nl_string(std::ofstream&) override;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  bool is_external_operator() override;
  std::string function_name;
  int external_function_index = -1;
  std::shared_ptr<Node>* operands;
  unsigned int nargs;
  void fill_expression(std::shared_ptr<Operator>* oper_array, int& oper_ndx) override;
};


class SubtractOperator: public BinaryOperator
{
public:
  SubtractOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "SubtractOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_subtract_operator() override;
};


class DivideOperator: public BinaryOperator
{
public:
  DivideOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "DivideOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_divide_operator() override;
};


class PowerOperator: public BinaryOperator
{
public:
  PowerOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "PowerOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_power_operator() override;
};


class NegationOperator: public UnaryOperator
{
public:
  NegationOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void print(std::string*) override;
  std::string name() override {return "NegationOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_negation_operator() override;
};


class ExpOperator: public UnaryOperator
{
public:
  ExpOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "ExpOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_exp_operator() override;
};


class LogOperator: public UnaryOperator
{
public:
  LogOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "LogOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_log_operator() override;
};


class Log10Operator: public UnaryOperator
{
public:
  Log10Operator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "Log10Operator";};
  void write_nl_string(std::ofstream&) override;
};


class SinOperator: public UnaryOperator
{
public:
  SinOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "SinOperator";};
  void write_nl_string(std::ofstream&) override;
};


class CosOperator: public UnaryOperator
{
public:
  CosOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "CosOperator";};
  void write_nl_string(std::ofstream&) override;
};


class TanOperator: public UnaryOperator
{
public:
  TanOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "TanOperator";};
  void write_nl_string(std::ofstream&) override;
};


class AsinOperator: public UnaryOperator
{
public:
  AsinOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "AsinOperator";};
  void write_nl_string(std::ofstream&) override;
};


class AcosOperator: public UnaryOperator
{
public:
  AcosOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "AcosOperator";};
  void write_nl_string(std::ofstream&) override;
};


class AtanOperator: public UnaryOperator
{
public:
  AtanOperator() = default;
  void evaluate(double* values) override;
  void print(std::string*) override;
  std::string name() override {return "AtanOperator";};
  void write_nl_string(std::ofstream&) override;
};


class PyomoExprTypes
{
public:
  PyomoExprTypes() {
    expr_type_map[int_] = 0;
    expr_type_map[float_] = 0;
    expr_type_map[ScalarVar] = 1;
    expr_type_map[_GeneralVarData] = 1;
    expr_type_map[ScalarParam] = 2;
    expr_type_map[_ParamData] = 2;
    expr_type_map[MonomialTermExpression] = 3;
    expr_type_map[ProductExpression] = 3;
    expr_type_map[NPV_ProductExpression] = 3;
    expr_type_map[SumExpression] = 4;
    expr_type_map[NPV_SumExpression] = 4;
    expr_type_map[NegationExpression] = 5;
    expr_type_map[NPV_NegationExpression] = 5;
    expr_type_map[ExternalFunctionExpression] = 6;
    expr_type_map[NPV_ExternalFunctionExpression] = 6;
    expr_type_map[PowExpression] = 7;
    expr_type_map[NPV_PowExpression] = 7;
    expr_type_map[DivisionExpression] = 8;
    expr_type_map[NPV_DivisionExpression] = 8;
    expr_type_map[UnaryFunctionExpression] = 9;
    expr_type_map[NPV_UnaryFunctionExpression] = 9;
    expr_type_map[LinearExpression] = 10;
    expr_type_map[_GeneralExpressionData] = 11;
    expr_type_map[ScalarExpression] = 11;
    expr_type_map[Integral] = 11;
    expr_type_map[ScalarIntegral] = 11;
    expr_type_map[NumericConstant] = 12;
  }
  ~PyomoExprTypes() = default;
  py::int_ ione = 1;
  py::float_ fone = 1.0;
  py::type int_ = py::type::of(ione);
  py::type float_ = py::type::of(fone);
  py::object ScalarParam = py::module_::import("pyomo.core.base.param").attr("ScalarParam");
  py::object _ParamData = py::module_::import("pyomo.core.base.param").attr("_ParamData");
  py::object ScalarVar = py::module_::import("pyomo.core.base.var").attr("ScalarVar");
  py::object _GeneralVarData = py::module_::import("pyomo.core.base.var").attr("_GeneralVarData");
  py::object numeric_expr = py::module_::import("pyomo.core.expr.numeric_expr");
  py::object NegationExpression = numeric_expr.attr("NegationExpression");
  py::object NPV_NegationExpression = numeric_expr.attr("NPV_NegationExpression");
  py::object ExternalFunctionExpression = numeric_expr.attr("ExternalFunctionExpression");
  py::object NPV_ExternalFunctionExpression = numeric_expr.attr("NPV_ExternalFunctionExpression");
  py::object PowExpression = numeric_expr.attr("PowExpression");
  py::object NPV_PowExpression = numeric_expr.attr("NPV_PowExpression");
  py::object ProductExpression = numeric_expr.attr("ProductExpression");
  py::object NPV_ProductExpression = numeric_expr.attr("NPV_ProductExpression");
  py::object MonomialTermExpression = numeric_expr.attr("MonomialTermExpression");
  py::object DivisionExpression = numeric_expr.attr("DivisionExpression");
  py::object NPV_DivisionExpression = numeric_expr.attr("NPV_DivisionExpression");
  py::object SumExpression = numeric_expr.attr("SumExpression");
  py::object NPV_SumExpression = numeric_expr.attr("NPV_SumExpression");
  py::object UnaryFunctionExpression = numeric_expr.attr("UnaryFunctionExpression");
  py::object NPV_UnaryFunctionExpression = numeric_expr.attr("NPV_UnaryFunctionExpression");
  py::object LinearExpression = numeric_expr.attr("LinearExpression");
  py::object NumericConstant = py::module_::import("pyomo.core.expr.numvalue").attr("NumericConstant");
  py::object expr_module = py::module_::import("pyomo.core.base.expression");
  py::object _GeneralExpressionData = expr_module.attr("_GeneralExpressionData");
  py::object ScalarExpression = expr_module.attr("ScalarExpression");
  py::object ScalarIntegral = py::module_::import("pyomo.dae.integral").attr("ScalarIntegral");
  py::object Integral = py::module_::import("pyomo.dae.integral").attr("Integral");
  py::object builtins = py::module_::import("builtins");
  py::object id = builtins.attr("id");
  py::object len = builtins.attr("len");
  py::dict expr_type_map;
};


std::vector<std::shared_ptr<Var> > create_vars(int n_vars);
std::vector<std::shared_ptr<Param> > create_params(int n_params);
std::vector<std::shared_ptr<Constant> > create_constants(int n_constants);
std::shared_ptr<ExpressionBase> appsi_expr_from_pyomo_expr(py::handle expr, py::handle var_map, py::handle param_map, PyomoExprTypes& expr_types);
std::vector<std::shared_ptr<ExpressionBase> > appsi_exprs_from_pyomo_exprs(py::list expr_list, py::dict var_map, py::dict param_map);
py::tuple prep_for_repn(py::handle expr, PyomoExprTypes& expr_types);
