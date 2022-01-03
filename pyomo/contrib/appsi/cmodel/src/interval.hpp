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
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace pybind11::literals;


extern double inf;

double _pow_with_inf(double x, double y);

void interval_add(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
void interval_sub(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
void interval_mul(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
void interval_inv(double xl, double xu, double* res_lb, double* res_ub, double feasibility_tol);
void interval_div(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub, double feasibility_tol);
void interval_power(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub, double feasibility_tol);
void interval_exp(double xl, double xu, double* res_lb, double* res_ub);
void interval_log(double xl, double xu, double* res_lb, double* res_ub);
void _inverse_power1(double zl, double zu, double yl, double yu, double orig_xl, double orig_xu, double* xl, double* xu, double feasibility_tol);
void _inverse_power2(double zl, double zu, double xl, double xu, double* yl, double* yu, double feasiblity_tol);
void interval_log10(double xl, double xu, double* res_lb, double* res_ub);
void interval_sin(double xl, double xu, double* res_lb, double* res_ub);
void interval_cos(double xl, double xu, double* res_lb, double* res_ub);
void interval_tan(double xl, double xu, double* res_lb, double* res_ub);
void interval_asin(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub, double feasibility_tol);
void interval_acos(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub, double feasibility_tol);
void interval_atan(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
std::pair<double, double> py_interval_add(double xl, double xu, double yl, double yu);
std::pair<double, double> py_interval_sub(double xl, double xu, double yl, double yu);
std::pair<double, double> py_interval_mul(double xl, double xu, double yl, double yu);
std::pair<double, double> py_interval_inv(double xl, double xu, double feasibility_tol);
std::pair<double, double> py_interval_div(double xl, double xu, double yl, double yu, double feasibility_tol);
std::pair<double, double> py_interval_power(double xl, double xu, double yl, double yu, double feasibility_tol);
std::pair<double, double> py_interval_exp(double xl, double xu);
std::pair<double, double> py_interval_log(double xl, double xu);
std::pair<double, double> _py_inverse_power1(double zl, double zu, double yl, double yu, double orig_xl, double orig_xu, double feasibility_tol);
std::pair<double, double> _py_inverse_power2(double zl, double zu, double xl, double xu, double feasibility_tol);
std::pair<double, double> py_interval_log10(double xl, double xu);
std::pair<double, double> py_interval_sin(double xl, double xu);
std::pair<double, double> py_interval_cos(double xl, double xu);
std::pair<double, double> py_interval_tan(double xl, double xu);
std::pair<double, double> py_interval_asin(double xl, double xu, double yl, double yu, double feasibility_tol);
std::pair<double, double> py_interval_acos(double xl, double xu, double yl, double yu, double feasibility_tol);
std::pair<double, double> py_interval_atan(double xl, double xu, double yl, double yu);
