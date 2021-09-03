#include "expression.hpp"


std::pair<double, double> add(double xl, double xu, double yl, double yu);
std::pair<double, double> sub(double xl, double xu, double yl, double yu);
std::pair<double, double> mul(double xl, double xu, double yl, double yu);
std::pair<double, double> inv(double xl, double xu, double feasibility_tol);
std::pair<double, double> div(double xl, double xu, double yl, double yu, double feasibility_tol);
std::pair<double, double> power(double xl, double xu, double yl, double yu, double feasibility_tol);
std::pair<double, double> inverse_power1(double zl, double zu, double yl, double yu, double orig_xl, double orig_xu, double feasibility_tol);
std::pair<double, double> inverse_power2(double zl, double zu, double xl, double xu, double feasibility_tol);
std::pair<double, double> exp(double xl, double xu);
std::pair<double, double> log(double xl, double xu);
std::pair<double, double> log10(double xl, double xu);
std::pair<double, double> sin(double xl, double xu);
std::pair<double, double> cos(double xl, double xu);
std::pair<double, double> tan(double xl, double xu);
std::pair<double, double> asin(double xl, double xu, double feasibility_tol);
std::pair<double, double> acos(double xl, double xu, double feasibility_tol);
std::pair<double, double> atan(double xl, double xu);
