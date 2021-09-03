#include "interval.hpp"


std::pair<double, double> add(double xl, double xu, double yl, double yu)
{
  return std::make_pair(xl + yl, xu + yu);
}


std::pair<double, double> sub(double xl, double xu, double yl, double yu)
{
  return std::make_pair(xl - yu, xu - yl);
}


std::pair<double, double> mul(double xl, double xu, double yl, double yu)
{
  std::vector<double> options;
  std::vector<double> x_bnds = {xl, xu};
  std::vector<double> y_bnds = {yl, yu};

  for (double &_x : x_bnds)
    {
      for (double &_y : y_bnds)
	{
	  if ((_x <= -inf) || (_x >= inf) || (_y <= -inf) || (_y >= inf))
	    {
	      options.push_back(-inf);
	      options.push_back(inf);
	    }
	  else
	    {
	      options.push_back(_x * _y);
	    }
	}
    }

  double lb = *std::min_element(options.begin(), options.end());
  double ub = *std::max_element(options.begin(), options.end());

  return std::make_pair(lb, ub);
}


std::pair<double, double> inv(double xl, double xu, double feasibility_tol)
{
  /*
    The case where xl is very slightly positive but should be very slightly negative (or xu is very slightly negative
    but should be very slightly positive) should not be an issue. Suppose xu is 2 and xl is 1e-15 but should be -1e-15.
    The bounds obtained from this function will be [0.5, 1e15] or [0.5, inf), depending on the value of
    feasibility_tol. The true bounds are (-inf, -1e15] U [0.5, inf), where U is union. The exclusion of (-inf, -1e15]
    should be acceptable. Additionally, it very important to return a non-negative interval when xl is non-negative.
  */

  double lb;
  double ub;
  
  if (xu - xl <= -feasibility_tol)
    {
      throw std::string("lower bound is greater than upper bound in inv; xl: " + std::to_string(xl) + "; xu: " + std::to_string(xu));
    }
  else if (xu <= 0 && 0 <= xl)
    {
      // This has to return -inf to inf because it could later be multiplied by 0
      lb = -inf;
      ub = inf;
    }
  else if (xl < 0 && 0 < xu)
    {
      lb = -inf;
      ub = inf;
    }
  else if (0 <= xl && xl <= feasibility_tol)
    {
      // xu must be strictly positive
      ub = inf;
      lb = 1.0 / xu;
    }
  else if (xl > feasibility_tol)
    {
      // xl and xu must be strictly positive
      ub = 1.0 / xl;
      lb = 1.0 / xu;
    }
  else if (-feasibility_tol <= xu && xu <= 0)
    {
      // xl must be strictly negative
      lb = -inf;
      ub = 1.0 / xl;
    }
  else if (xu < -feasibility_tol)
    {
      // xl and xu must be strictly negative
      ub = 1.0 / xl;
      lb = 1.0 / xu;
    }
  else
    {
      // everything else
      lb = -inf;
      ub = inf;
    }
  return std::make_pari(lb, ub);
}


std::pair<double, double> div(double xl, double xu, double yl, double yu, double feasibility_tol)
{
  double lb;
  double ub;
}


std::pair<double, double> power(double xl, double xu, double yl, double yu, double feasibility_tol)
{
}


std::pair<double, double> inverse_power1(double zl, double zu, double yl, double yu, double orig_xl, double orig_xu, double feasibility_tol)
{
}


std::pair<double, double> inverse_power2(double zl, double zu, double xl, double xu, double feasibility_tol)
{
}


std::pair<double, double> exp(double xl, double xu)
{
}


std::pair<double, double> log(double xl, double xu)
{
}


std::pair<double, double> log10(double xl, double xu)
{
}


std::pair<double, double> sin(double xl, double xu)
{
}


std::pair<double, double> cos(double xl, double xu)
{
}


std::pair<double, double> tan(double xl, double xu)
{
}


std::pair<double, double> asin(double xl, double xu, double feasibility_tol)
{
}


std::pair<double, double> acos(double xl, double xu, double feasibility_tol)
{
}


std::pair<double, double> atan(double xl, double xu)
{
}

