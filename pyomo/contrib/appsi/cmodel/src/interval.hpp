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


extern double inf;


void add(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
void sub(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
void mul(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub);
void div(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub, double feasibility_tol);
