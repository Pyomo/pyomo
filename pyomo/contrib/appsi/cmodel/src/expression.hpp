/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 * National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#ifndef EXPRESSION_HEADER
#define EXPRESSION_HEADER

#include "interval.hpp"
#include <mutex>

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
class DivideOperator;
class PowerOperator;
class NegationOperator;
class ExpOperator;
class LogOperator;
class AbsOperator;
class ExternalOperator;
class PyomoExprTypes;

extern double inf;

class Node : public std::enable_shared_from_this<Node> {
public:
  Node() = default;
  virtual ~Node() = default;
  virtual bool is_variable_type() { return false; }
  virtual bool is_param_type() { return false; }
  virtual bool is_expression_type() { return false; }
  virtual bool is_operator_type() { return false; }
  virtual bool is_constant_type() { return false; }
  virtual bool is_leaf() { return false; }
  virtual bool is_binary_operator() { return false; }
  virtual bool is_unary_operator() { return false; }
  virtual bool is_linear_operator() { return false; }
  virtual bool is_sum_operator() { return false; }
  virtual bool is_multiply_operator() { return false; }
  virtual bool is_divide_operator() { return false; }
  virtual bool is_power_operator() { return false; }
  virtual bool is_negation_operator() { return false; }
  virtual bool is_exp_operator() { return false; }
  virtual bool is_log_operator() { return false; }
  virtual bool is_abs_operator() { return false; }
  virtual bool is_sqrt_operator() { return false; }
  virtual bool is_external_operator() { return false; }
  virtual double get_value_from_array(double *) = 0;
  virtual int get_degree_from_array(int *) = 0;
  virtual std::string get_string_from_array(std::string *) = 0;
  virtual void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) = 0;
  virtual void write_nl_string(std::ofstream &) = 0;
  virtual void fill_expression(std::shared_ptr<Operator> *oper_array,
                               int &oper_ndx) = 0;
  virtual double get_lb_from_array(double *lbs) = 0;
  virtual double get_ub_from_array(double *ubs) = 0;
  virtual void
  set_bounds_in_array(double new_lb, double new_ub, double *lbs, double *ubs,
                      double feasibility_tol, double integer_tol,
                      double improvement_tol,
                      std::set<std::shared_ptr<Var>> &improved_vars) = 0;
};

class ExpressionBase : public Node {
public:
  ExpressionBase() = default;
  virtual double evaluate() = 0;
  virtual std::string __str__() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Var>>>
  identify_variables() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator>>>
  identify_external_operators() = 0;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Node>>>
  get_prefix_notation() = 0;
  std::shared_ptr<ExpressionBase> shared_from_this() {
    return std::static_pointer_cast<ExpressionBase>(Node::shared_from_this());
  }
  void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) override {
    ;
  }
};

class Leaf : public ExpressionBase {
public:
  Leaf() = default;
  Leaf(double value) : value(value) {}
  virtual ~Leaf() = default;
  double value = 0.0;
  bool is_leaf() override;
  double evaluate() override;
  double get_value_from_array(double *) override;
  std::string get_string_from_array(std::string *) override;
  std::shared_ptr<std::vector<std::shared_ptr<Node>>>
  get_prefix_notation() override;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
  double get_lb_from_array(double *lbs) override;
  double get_ub_from_array(double *ubs) override;
  void
  set_bounds_in_array(double new_lb, double new_ub, double *lbs, double *ubs,
                      double feasibility_tol, double integer_tol,
                      double improvement_tol,
                      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class Constant : public Leaf {
public:
  Constant() = default;
  Constant(double value) : Leaf(value) {}
  bool is_constant_type() override;
  std::string __str__() override;
  int get_degree_from_array(int *) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>>
  identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator>>>
  identify_external_operators() override;
  void write_nl_string(std::ofstream &) override;
};

enum Domain { continuous, binary, integers };

class Var : public Leaf {
public:
  Var() = default;
  Var(double val) : Leaf(val) {}
  Var(std::string _name) : name(_name) {}
  Var(std::string _name, double val) : Leaf(val), name(_name) {}
  std::string name = "v";
  std::string __str__() override;
  std::shared_ptr<ExpressionBase> lb;
  std::shared_ptr<ExpressionBase> ub;
  int index = -1;
  bool fixed = false;
  double domain_lb = -inf;
  double domain_ub = inf;
  Domain domain = continuous;
  bool is_variable_type() override;
  int get_degree_from_array(int *) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>>
  identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator>>>
  identify_external_operators() override;
  void write_nl_string(std::ofstream &) override;
  std::shared_ptr<Var> shared_from_this() {
    return std::static_pointer_cast<Var>(Node::shared_from_this());
  }
  double get_lb();
  double get_ub();
  Domain get_domain();
  double get_lb_from_array(double *lbs) override;
  double get_ub_from_array(double *ubs) override;
  void
  set_bounds_in_array(double new_lb, double new_ub, double *lbs, double *ubs,
                      double feasibility_tol, double integer_tol,
                      double improvement_tol,
                      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class Param : public Leaf {
public:
  Param() = default;
  Param(double val) : Leaf(val) {}
  Param(std::string _name) : name(_name) {}
  Param(std::string _name, double val) : Leaf(val), name(_name) {}
  std::string name = "p";
  std::string __str__() override;
  bool is_param_type() override;
  int get_degree_from_array(int *) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>>
  identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator>>>
  identify_external_operators() override;
  void write_nl_string(std::ofstream &) override;
};

class Expression : public ExpressionBase {
public:
  Expression(int _n_operators) : ExpressionBase() {
    operators = new std::shared_ptr<Operator>[_n_operators];
    n_operators = _n_operators;
  }
  ~Expression() { delete[] operators; }
  std::string __str__() override;
  bool is_expression_type() override;
  double evaluate() override;
  double get_value_from_array(double *) override;
  int get_degree_from_array(int *) override;
  std::shared_ptr<std::vector<std::shared_ptr<Var>>>
  identify_variables() override;
  std::shared_ptr<std::vector<std::shared_ptr<ExternalOperator>>>
  identify_external_operators() override;
  std::string get_string_from_array(std::string *) override;
  std::shared_ptr<std::vector<std::shared_ptr<Node>>>
  get_prefix_notation() override;
  void write_nl_string(std::ofstream &) override;
  std::vector<std::shared_ptr<Operator>> get_operators();
  std::shared_ptr<Operator> *operators;
  unsigned int n_operators;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol, double integer_tol);
  void propagate_bounds_backward(double *lbs, double *ubs,
                                 double feasibility_tol, double integer_tol,
                                 double improvement_tol,
                                 std::set<std::shared_ptr<Var>> &improved_vars);
  double get_lb_from_array(double *lbs) override;
  double get_ub_from_array(double *ubs) override;
  void
  set_bounds_in_array(double new_lb, double new_ub, double *lbs, double *ubs,
                      double feasibility_tol, double integer_tol,
                      double improvement_tol,
                      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class Operator : public Node {
public:
  Operator() = default;
  int index = 0;
  virtual void evaluate(double *values) = 0;
  virtual void propagate_degree_forward(int *degrees, double *values) = 0;
  virtual void
  identify_variables(std::set<std::shared_ptr<Node>> &,
                     std::shared_ptr<std::vector<std::shared_ptr<Var>>>) = 0;
  std::shared_ptr<Operator> shared_from_this() {
    return std::static_pointer_cast<Operator>(Node::shared_from_this());
  }
  bool is_operator_type() override;
  double get_value_from_array(double *) override;
  int get_degree_from_array(int *) override;
  std::string get_string_from_array(std::string *) override;
  virtual void print(std::string *) = 0;
  virtual std::string name() = 0;
  virtual void propagate_bounds_forward(double *lbs, double *ubs,
                                        double feasibility_tol,
                                        double integer_tol);
  virtual void
  propagate_bounds_backward(double *lbs, double *ubs, double feasibility_tol,
                            double integer_tol, double improvement_tol,
                            std::set<std::shared_ptr<Var>> &improved_vars);
  double get_lb_from_array(double *lbs) override;
  double get_ub_from_array(double *ubs) override;
  void
  set_bounds_in_array(double new_lb, double new_ub, double *lbs, double *ubs,
                      double feasibility_tol, double integer_tol,
                      double improvement_tol,
                      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class BinaryOperator : public Operator {
public:
  BinaryOperator() = default;
  virtual ~BinaryOperator() = default;
  void identify_variables(
      std::set<std::shared_ptr<Node>> &,
      std::shared_ptr<std::vector<std::shared_ptr<Var>>>) override;
  std::shared_ptr<Node> operand1;
  std::shared_ptr<Node> operand2;
  void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) override;
  bool is_binary_operator() override;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
};

class UnaryOperator : public Operator {
public:
  UnaryOperator() = default;
  virtual ~UnaryOperator() = default;
  void identify_variables(
      std::set<std::shared_ptr<Node>> &,
      std::shared_ptr<std::vector<std::shared_ptr<Var>>>) override;
  std::shared_ptr<Node> operand;
  void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) override;
  bool is_unary_operator() override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
};

class LinearOperator : public Operator {
public:
  LinearOperator(int _nterms) {
    variables = new std::shared_ptr<Var>[_nterms];
    coefficients = new std::shared_ptr<ExpressionBase>[_nterms];
    nterms = _nterms;
  }
  ~LinearOperator() {
    delete[] variables;
    delete[] coefficients;
  }
  void identify_variables(
      std::set<std::shared_ptr<Node>> &,
      std::shared_ptr<std::vector<std::shared_ptr<Var>>>) override;
  std::shared_ptr<Var> *variables;
  std::shared_ptr<ExpressionBase> *coefficients;
  std::shared_ptr<ExpressionBase> constant = std::make_shared<Constant>(0);
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "LinearOperator"; };
  void write_nl_string(std::ofstream &) override;
  void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) override;
  bool is_linear_operator() override;
  unsigned int nterms;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class SumOperator : public Operator {
public:
  SumOperator(int _nargs) {
    operands = new std::shared_ptr<Node>[_nargs];
    nargs = _nargs;
  }
  ~SumOperator() { delete[] operands; }
  void identify_variables(
      std::set<std::shared_ptr<Node>> &,
      std::shared_ptr<std::vector<std::shared_ptr<Var>>>) override;
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "SumOperator"; };
  void write_nl_string(std::ofstream &) override;
  void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) override;
  bool is_sum_operator() override;
  std::shared_ptr<Node> *operands;
  unsigned int nargs;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class MultiplyOperator : public BinaryOperator {
public:
  MultiplyOperator() = default;
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "MultiplyOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_multiply_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class ExternalOperator : public Operator {
public:
  ExternalOperator(int _nargs) {
    operands = new std::shared_ptr<Node>[_nargs];
    nargs = _nargs;
  }
  ~ExternalOperator() { delete[] operands; }
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "ExternalOperator"; };
  void write_nl_string(std::ofstream &) override;
  void fill_prefix_notation_stack(
      std::shared_ptr<std::vector<std::shared_ptr<Node>>> stack) override;
  void identify_variables(
      std::set<std::shared_ptr<Node>> &,
      std::shared_ptr<std::vector<std::shared_ptr<Var>>>) override;
  bool is_external_operator() override;
  std::string function_name;
  int external_function_index = -1;
  std::shared_ptr<Node> *operands;
  unsigned int nargs;
  void fill_expression(std::shared_ptr<Operator> *oper_array,
                       int &oper_ndx) override;
};

class DivideOperator : public BinaryOperator {
public:
  DivideOperator() = default;
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "DivideOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_divide_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class PowerOperator : public BinaryOperator {
public:
  PowerOperator() = default;
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "PowerOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_power_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class NegationOperator : public UnaryOperator {
public:
  NegationOperator() = default;
  void evaluate(double *values) override;
  void propagate_degree_forward(int *degrees, double *values) override;
  void print(std::string *) override;
  std::string name() override { return "NegationOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_negation_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class ExpOperator : public UnaryOperator {
public:
  ExpOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "ExpOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_exp_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class LogOperator : public UnaryOperator {
public:
  LogOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "LogOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_log_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class AbsOperator : public UnaryOperator {
public:
  AbsOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "AbsOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_abs_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class SqrtOperator : public UnaryOperator {
public:
  SqrtOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "SqrtOperator"; };
  void write_nl_string(std::ofstream &) override;
  bool is_sqrt_operator() override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class Log10Operator : public UnaryOperator {
public:
  Log10Operator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "Log10Operator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class SinOperator : public UnaryOperator {
public:
  SinOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "SinOperator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class CosOperator : public UnaryOperator {
public:
  CosOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "CosOperator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class TanOperator : public UnaryOperator {
public:
  TanOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "TanOperator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class AsinOperator : public UnaryOperator {
public:
  AsinOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "AsinOperator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class AcosOperator : public UnaryOperator {
public:
  AcosOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "AcosOperator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

class AtanOperator : public UnaryOperator {
public:
  AtanOperator() = default;
  void evaluate(double *values) override;
  void print(std::string *) override;
  std::string name() override { return "AtanOperator"; };
  void write_nl_string(std::ofstream &) override;
  void propagate_bounds_forward(double *lbs, double *ubs,
                                double feasibility_tol,
                                double integer_tol) override;
  void propagate_bounds_backward(
      double *lbs, double *ubs, double feasibility_tol, double integer_tol,
      double improvement_tol,
      std::set<std::shared_ptr<Var>> &improved_vars) override;
};

enum ExprType {
  py_float = 0,
  var = 1,
  param = 2,
  product = 3,
  sum = 4,
  negation = 5,
  external_func = 6,
  power = 7,
  division = 8,
  unary_func = 9,
  linear = 10,
  named_expr = 11,
  numeric_constant = 12,
  pyomo_unit = 13,
  unary_abs = 14
};

class PyomoExprTypes {
public:
  PyomoExprTypes() {
    expr_type_map[int_] = py_float;
    expr_type_map[float_] = py_float;
    expr_type_map[np_int16] = py_float;
    expr_type_map[np_int32] = py_float;
    expr_type_map[np_int64] = py_float;
    expr_type_map[np_longlong] = py_float;
    expr_type_map[np_uint16] = py_float;
    expr_type_map[np_uint32] = py_float;
    expr_type_map[np_uint64] = py_float;
    expr_type_map[np_ulonglong] = py_float;
    expr_type_map[np_float16] = py_float;
    expr_type_map[np_float32] = py_float;
    expr_type_map[np_float64] = py_float;
    expr_type_map[ScalarVar] = var;
    expr_type_map[VarData] = var;
    expr_type_map[AutoLinkedBinaryVar] = var;
    expr_type_map[ScalarParam] = param;
    expr_type_map[ParamData] = param;
    expr_type_map[MonomialTermExpression] = product;
    expr_type_map[ProductExpression] = product;
    expr_type_map[NPV_ProductExpression] = product;
    expr_type_map[SumExpression] = sum;
    expr_type_map[NPV_SumExpression] = sum;
    expr_type_map[NegationExpression] = negation;
    expr_type_map[NPV_NegationExpression] = negation;
    expr_type_map[ExternalFunctionExpression] = external_func;
    expr_type_map[NPV_ExternalFunctionExpression] = external_func;
    expr_type_map[PowExpression] = power;
    expr_type_map[NPV_PowExpression] = power;
    expr_type_map[DivisionExpression] = division;
    expr_type_map[NPV_DivisionExpression] = division;
    expr_type_map[UnaryFunctionExpression] = unary_func;
    expr_type_map[NPV_UnaryFunctionExpression] = unary_func;
    expr_type_map[LinearExpression] = linear;
    expr_type_map[ExpressionData] = named_expr;
    expr_type_map[ScalarExpression] = named_expr;
    expr_type_map[Integral] = named_expr;
    expr_type_map[ScalarIntegral] = named_expr;
    expr_type_map[NumericConstant] = numeric_constant;
    expr_type_map[_PyomoUnit] = pyomo_unit;
    expr_type_map[AbsExpression] = unary_abs;
    expr_type_map[NPV_AbsExpression] = unary_abs;
  }
  ~PyomoExprTypes() = default;
  py::int_ ione = 1;
  py::float_ fone = 1.0;
  py::type int_ = py::type::of(ione);
  py::type float_ = py::type::of(fone);
  py::object np = py::module_::import("numpy");
  py::type np_int16 = np.attr("int16");
  py::type np_int32 = np.attr("int32");
  py::type np_int64 = np.attr("int64");
  py::type np_longlong = np.attr("longlong");
  py::type np_uint16 = np.attr("uint16");
  py::type np_uint32 = np.attr("uint32");
  py::type np_uint64 = np.attr("uint64");
  py::type np_ulonglong = np.attr("ulonglong");
  py::type np_float16 = np.attr("float16");
  py::type np_float32 = np.attr("float32");
  py::type np_float64 = np.attr("float64");
  py::object ScalarParam =
      py::module_::import("pyomo.core.base.param").attr("ScalarParam");
  py::object ParamData =
      py::module_::import("pyomo.core.base.param").attr("ParamData");
  py::object ScalarVar =
      py::module_::import("pyomo.core.base.var").attr("ScalarVar");
  py::object VarData =
      py::module_::import("pyomo.core.base.var").attr("VarData");
  py::object AutoLinkedBinaryVar =
      py::module_::import("pyomo.gdp.disjunct").attr("AutoLinkedBinaryVar");
  py::object numeric_expr = py::module_::import("pyomo.core.expr.numeric_expr");
  py::object NegationExpression = numeric_expr.attr("NegationExpression");
  py::object NPV_NegationExpression =
      numeric_expr.attr("NPV_NegationExpression");
  py::object ExternalFunctionExpression =
      numeric_expr.attr("ExternalFunctionExpression");
  py::object NPV_ExternalFunctionExpression =
      numeric_expr.attr("NPV_ExternalFunctionExpression");
  py::object PowExpression = numeric_expr.attr("PowExpression");
  py::object NPV_PowExpression = numeric_expr.attr("NPV_PowExpression");
  py::object ProductExpression = numeric_expr.attr("ProductExpression");
  py::object NPV_ProductExpression = numeric_expr.attr("NPV_ProductExpression");
  py::object MonomialTermExpression =
      numeric_expr.attr("MonomialTermExpression");
  py::object DivisionExpression = numeric_expr.attr("DivisionExpression");
  py::object NPV_DivisionExpression =
      numeric_expr.attr("NPV_DivisionExpression");
  py::object SumExpression = numeric_expr.attr("SumExpression");
  py::object NPV_SumExpression = numeric_expr.attr("NPV_SumExpression");
  py::object UnaryFunctionExpression =
      numeric_expr.attr("UnaryFunctionExpression");
  py::object AbsExpression = numeric_expr.attr("AbsExpression");
  py::object NPV_AbsExpression = numeric_expr.attr("NPV_AbsExpression");
  py::object NPV_UnaryFunctionExpression =
      numeric_expr.attr("NPV_UnaryFunctionExpression");
  py::object LinearExpression = numeric_expr.attr("LinearExpression");
  py::object NumericConstant =
      py::module_::import("pyomo.core.expr.numvalue").attr("NumericConstant");
  py::object expr_module = py::module_::import("pyomo.core.base.expression");
  py::object ExpressionData =
      expr_module.attr("ExpressionData");
  py::object ScalarExpression = expr_module.attr("ScalarExpression");
  py::object ScalarIntegral =
      py::module_::import("pyomo.dae.integral").attr("ScalarIntegral");
  py::object Integral =
      py::module_::import("pyomo.dae.integral").attr("Integral");
  py::object _PyomoUnit =
      py::module_::import("pyomo.core.base.units_container").attr("_PyomoUnit");
  py::object builtins = py::module_::import("builtins");
  py::object id = builtins.attr("id");
  py::object len = builtins.attr("len");
  py::dict expr_type_map;
};

std::vector<std::shared_ptr<Var>> create_vars(int n_vars);
std::vector<std::shared_ptr<Param>> create_params(int n_params);
std::vector<std::shared_ptr<Constant>> create_constants(int n_constants);
std::shared_ptr<ExpressionBase>
appsi_expr_from_pyomo_expr(py::handle expr, py::handle var_map,
                           py::handle param_map, PyomoExprTypes &expr_types);
std::vector<std::shared_ptr<ExpressionBase>>
appsi_exprs_from_pyomo_exprs(py::list expr_list, py::dict var_map,
                             py::dict param_map);
py::tuple prep_for_repn(py::handle expr, PyomoExprTypes &expr_types);

void process_pyomo_vars(PyomoExprTypes &expr_types, py::list pyomo_vars,
                        py::dict var_map, py::dict param_map,
                        py::dict var_attrs, py::dict rev_var_map,
                        py::bool_ _set_name, py::handle symbol_map,
                        py::handle labeler, py::bool_ _update);

#endif
