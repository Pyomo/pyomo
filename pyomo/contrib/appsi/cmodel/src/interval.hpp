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

#ifndef INTERVAL_HEADER
#define INTERVAL_HEADER

#include <iostream>
#include <list>
#include <vector>
#define _USE_MATH_DEFINES
#include "common.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>

extern double inf;

double _pow_with_inf(double x, double y);

void interval_add(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub);
void interval_sub(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub);
void interval_mul(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub);
void interval_inv(double xl, double xu, double *res_lb, double *res_ub,
                  double feasibility_tol);
void interval_div(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub, double feasibility_tol);
void interval_power(double xl, double xu, double yl, double yu, double *res_lb,
                    double *res_ub, double feasibility_tol);
void interval_exp(double xl, double xu, double *res_lb, double *res_ub);
void interval_log(double xl, double xu, double *res_lb, double *res_ub);
void interval_abs(double xl, double xu, double *res_lb, double *res_ub);
void _inverse_abs(double zl, double zu, double *xl, double *xu);
void _inverse_power1(double zl, double zu, double yl, double yu, double orig_xl,
                     double orig_xu, double *xl, double *xu,
                     double feasibility_tol);
void _inverse_power2(double zl, double zu, double xl, double xu, double *yl,
                     double *yu, double feasiblity_tol);
void interval_log10(double xl, double xu, double *res_lb, double *res_ub);
void interval_sin(double xl, double xu, double *res_lb, double *res_ub);
void interval_cos(double xl, double xu, double *res_lb, double *res_ub);
void interval_tan(double xl, double xu, double *res_lb, double *res_ub);
void interval_asin(double xl, double xu, double yl, double yu, double *res_lb,
                   double *res_ub, double feasibility_tol);
void interval_acos(double xl, double xu, double yl, double yu, double *res_lb,
                   double *res_ub, double feasibility_tol);
void interval_atan(double xl, double xu, double yl, double yu, double *res_lb,
                   double *res_ub);
std::pair<double, double> py_interval_add(double xl, double xu, double yl,
                                          double yu);
std::pair<double, double> py_interval_sub(double xl, double xu, double yl,
                                          double yu);
std::pair<double, double> py_interval_mul(double xl, double xu, double yl,
                                          double yu);
std::pair<double, double> py_interval_inv(double xl, double xu,
                                          double feasibility_tol);
std::pair<double, double> py_interval_div(double xl, double xu, double yl,
                                          double yu, double feasibility_tol);
std::pair<double, double> py_interval_power(double xl, double xu, double yl,
                                            double yu, double feasibility_tol);
std::pair<double, double> py_interval_exp(double xl, double xu);
std::pair<double, double> py_interval_log(double xl, double xu);
std::pair<double, double> py_interval_abs(double xl, double xu);
std::pair<double, double> _py_inverse_abs(double zl, double zu);
std::pair<double, double> _py_inverse_power1(double zl, double zu, double yl,
                                             double yu, double orig_xl,
                                             double orig_xu,
                                             double feasibility_tol);
std::pair<double, double> _py_inverse_power2(double zl, double zu, double xl,
                                             double xu, double feasibility_tol);
std::pair<double, double> py_interval_log10(double xl, double xu);
std::pair<double, double> py_interval_sin(double xl, double xu);
std::pair<double, double> py_interval_cos(double xl, double xu);
std::pair<double, double> py_interval_tan(double xl, double xu);
std::pair<double, double> py_interval_asin(double xl, double xu, double yl,
                                           double yu, double feasibility_tol);
std::pair<double, double> py_interval_acos(double xl, double xu, double yl,
                                           double yu, double feasibility_tol);
std::pair<double, double> py_interval_atan(double xl, double xu, double yl,
                                           double yu);

class IntervalException : public std::exception {
public:
  explicit IntervalException(std::string m) : message{m} {}
  const char *what() const noexcept override { return message.c_str(); }
  std::string message = "";
};

class InfeasibleConstraintException : public std::exception {
public:
  explicit InfeasibleConstraintException(std::string m) : message{m} {}
  const char *what() const noexcept override { return message.c_str(); }
  std::string message = "";
};

#endif
