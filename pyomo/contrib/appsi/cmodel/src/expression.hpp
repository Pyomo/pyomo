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
class AddOperator;
class SubtractOperator;
class DivideOperator;
class PowerOperator;
class NegationOperator;
class ExpOperator;
class LogOperator;
class ExternalOperator;
class Repn;
class TmpRepn;


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
  virtual bool is_external() {return false;}
  virtual bool is_binary_operator() {return false;}
  virtual bool is_unary_operator() {return false;}
  virtual bool is_linear_operator() {return false;}
  virtual bool is_sum_operator() {return false;}
  virtual bool is_multiply_operator() {return false;}
  virtual bool is_add_operator() {return false;}
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
  virtual void set_repn_info(bool*, bool) = 0;
  virtual void set_repn_info(int*, int) = 0;
  virtual bool get_accounted_for_from_array(bool*) = 0;
};


class ExpressionBase: public Node
{
public:
  ExpressionBase() = default;
  virtual double evaluate() = 0;
  virtual std::shared_ptr<Repn> generate_repn() = 0;
  std::shared_ptr<ExpressionBase> operator+(ExpressionBase&);
  std::shared_ptr<ExpressionBase> operator*(ExpressionBase&);
  std::shared_ptr<ExpressionBase> operator-(ExpressionBase&);
  std::shared_ptr<ExpressionBase> operator/(ExpressionBase&);
  std::shared_ptr<ExpressionBase> __pow__(ExpressionBase&);
  std::shared_ptr<ExpressionBase> operator-();

  std::shared_ptr<ExpressionBase> operator+(double);
  std::shared_ptr<ExpressionBase> operator*(double);
  std::shared_ptr<ExpressionBase> operator-(double);
  std::shared_ptr<ExpressionBase> operator/(double);
  std::shared_ptr<ExpressionBase> __pow__(double);

  std::shared_ptr<ExpressionBase> __radd__(double);
  std::shared_ptr<ExpressionBase> __rmul__(double);
  std::shared_ptr<ExpressionBase> __rsub__(double);
  std::shared_ptr<ExpressionBase> __rdiv__(double);
  std::shared_ptr<ExpressionBase> __rtruediv__(double);
  std::shared_ptr<ExpressionBase> __rpow__(double);

  virtual std::string __str__() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Var> > > identify_variables() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator> > > identify_external_operators() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Node> > > get_prefix_notation() = 0;
  virtual std::shared_ptr<ExpressionBase> distribute_products() = 0;

  std::shared_ptr<ExpressionBase> shared_from_this() {return std::static_pointer_cast<ExpressionBase>(Node::shared_from_this());}

  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override {;}
};


std::shared_ptr<ExpressionBase> appsi_exp(std::shared_ptr<ExpressionBase> n);
std::shared_ptr<ExpressionBase> appsi_log(std::shared_ptr<ExpressionBase> n);
std::shared_ptr<ExpressionBase> appsi_sum(std::vector<std::shared_ptr<ExpressionBase> > exprs_to_sum);
std::shared_ptr<ExpressionBase> external_helper(std::string function_name, std::vector<std::shared_ptr<ExpressionBase> > operands);


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
  bool get_accounted_for_from_array(bool*) override;
  std::string get_string_from_array(std::string*) override;
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > get_prefix_notation() override;
  void set_repn_info(bool*, bool) override;
  void set_repn_info(int*, int) override;
  std::shared_ptr<ExpressionBase> distribute_products() override;
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
  std::shared_ptr<Repn> generate_repn() override;
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
  std::shared_ptr<Repn> generate_repn() override;
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
  std::shared_ptr<Repn> generate_repn() override;
};


class Expression: public ExpressionBase
{
public:
  Expression() = default;
  unsigned int n_operators = 0;
  std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators = std::make_shared<std::vector<std::shared_ptr<Operator> > >();
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
  void set_repn_info(bool*, bool) override;
  void set_repn_info(int*, int) override;
  bool get_accounted_for_from_array(bool*) override;
  std::shared_ptr<Repn> generate_repn() override;
  std::shared_ptr<ExpressionBase> distribute_products() override;
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
  virtual void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) = 0;
  virtual void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) = 0;
  void set_repn_info(bool*, bool) override;
  void set_repn_info(int*, int) override;
  bool get_accounted_for_from_array(bool*) override;
  virtual void distribute_products(std::shared_ptr<std::vector<std::shared_ptr<Operator> > > new_operators, std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process, std::shared_ptr<std::unordered_set<std::shared_ptr<Node> > > already_processed, std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced);
  virtual std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) = 0;
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
};


class LinearOperator: public Operator
{
public:
  LinearOperator() = default;
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var> > > variables;
  std::shared_ptr<std::vector<std::shared_ptr<ExpressionBase> > > coefficients;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "LinearOperator";};
  void write_nl_string(std::ofstream&) override;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  bool is_linear_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class SumOperator: public Operator
{
public:
  SumOperator() = default;
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > operands = std::make_shared<std::vector<std::shared_ptr<Node> > >();
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "SumOperator";};
  void write_nl_string(std::ofstream&) override;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  bool is_sum_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class MultiplyOperator: public BinaryOperator
{
public:
  MultiplyOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "MultiplyOperator";};
  void write_nl_string(std::ofstream&) override;
  void distribute_products(std::shared_ptr<std::vector<std::shared_ptr<Operator> > > new_operators, std::shared_ptr<std::vector<std::shared_ptr<Operator> > > operators_to_process, std::shared_ptr<std::unordered_set<std::shared_ptr<Node> > > already_processed, std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
  bool is_multiply_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class ExternalOperator: public Operator
{
public:
  ExternalOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "ExternalOperator";};
  void write_nl_string(std::ofstream&) override;
  void fill_prefix_notation_stack(std::shared_ptr<std::vector<std::shared_ptr<Node> > > stack) override;
  void identify_variables(std::set<std::shared_ptr<Node> >&) override;
  bool is_external() override;
  std::shared_ptr<std::vector<std::shared_ptr<Node> > > operands = std::make_shared<std::vector<std::shared_ptr<Node> > >();
  std::string function_name;
  int external_function_index = -1;
  bool is_external_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class AddOperator: public BinaryOperator
{
public:
  AddOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "AddOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_add_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class SubtractOperator: public BinaryOperator
{
public:
  SubtractOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "SubtractOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_subtract_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class DivideOperator: public BinaryOperator
{
public:
  DivideOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "DivideOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_divide_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class PowerOperator: public BinaryOperator
{
public:
  PowerOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "PowerOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_power_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class NegationOperator: public UnaryOperator
{
public:
  NegationOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "NegationOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_negation_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class ExpOperator: public UnaryOperator
{
public:
  ExpOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "ExpOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_exp_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class LogOperator: public UnaryOperator
{
public:
  LogOperator() = default;
  void evaluate(double* values) override;
  void propagate_degree_forward(int* degrees, double* values) override;
  void propagate_repn_info_backward(int* degrees, bool* push, bool* negate) override;
  void generate_repn(std::shared_ptr<TmpRepn>, int*, bool*, bool*, bool*, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>, std::shared_ptr<SumOperator>) override;
  void print(std::string*) override;
  std::string name() override {return "LogOperator";};
  void write_nl_string(std::ofstream&) override;
  bool is_log_operator() override;
  std::shared_ptr<Operator> replace_operands(std::shared_ptr<std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node> > > needs_replaced) override;
};


class Repn
{
public:
  Repn() = default;
  ~Repn() = default;
  std::shared_ptr<ExpressionBase> constant;
  std::shared_ptr<ExpressionBase> linear;
  std::shared_ptr<ExpressionBase> quadratic;
  std::shared_ptr<ExpressionBase> nonlinear;
  void reset_with_constants();
  std::string __str__();
};


class TmpRepn
{
public:
  TmpRepn() = default;
  ~TmpRepn() = default;
  std::shared_ptr<Expression> constant;
  std::shared_ptr<Expression> linear;
  std::shared_ptr<Expression> quadratic;
  std::shared_ptr<Expression> nonlinear;
  void reset_with_expressions();
};


std::vector<std::shared_ptr<Var> > create_vars(int n_vars);
std::vector<std::shared_ptr<Param> > create_params(int n_params);
std::vector<std::shared_ptr<Constant> > create_constants(int n_constants);
std::vector<std::shared_ptr<Repn> > generate_repns(std::vector<std::shared_ptr<ExpressionBase> > exprs);

std::vector<py::object> generate_prefix_notation(py::object);
