#include "common.hpp"

double inf = py::module_::import("math").attr("inf").cast<double>();
