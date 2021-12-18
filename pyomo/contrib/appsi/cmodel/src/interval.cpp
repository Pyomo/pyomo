#include "interval.hpp"


void add(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub)
{
  if (xl <= -inf || yl <= -inf)
    {
      *res_lb = -inf;
    }
  else if (xl >= inf || yl >= inf)
    {
      *res_lb = inf;
    }
  else
    {
      *res_lb = xl + yl;
    }

  if (xu >= inf || yu >= inf)
    {
      *res_ub = inf;
    }
  else if (xu <= -inf || yu <= -inf)
    {
      *res_ub = -inf;
    }
  else
    {
      *res_ub = xu + yu;
    }
}


void sub(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub)
{
  if (xl <= -inf || yu >= inf)
    {
      *res_lb = -inf;
    }
  else if (xl >= inf || yu <= -inf)
    {
      *res_lb = inf;
    }
  else
    {
      *res_lb = xl - yu;
    }

  if (xu >= inf || yl <= -inf)
    {
      *res_ub = inf;
    }
  else if (xu <= -inf || yl >= inf)
    {
      *res_ub = -inf;
    }
  else
    {
      *res_ub = xu - yl;
    }
}


void _get_mul_option(double x, double y, double* option_lb, double* option_ub)
{
  if (x <= -inf)
    {
      if (y < 0)
	{
	  *option_lb = inf;
	  *option_ub = inf;
	}
      else if (y > 0)
	{
	  *option_lb = -inf;
	  *option_ub = -inf;
	}
      else
	{
	  *option_lb = -inf;
	  *option_ub = inf;
	}
    }
  else if (x >= inf)
    {
      if (y < 0)
	{
	  *option_lb = -inf;
	  *option_ub = -inf;
	}
      else if (y > 0)
	{
	  *option_lb = inf;
	  *option_ub = inf;
	}
      else
	{
	  *option_lb = -inf;
	  *option_ub = inf;
	}
    }
  else if (y <= -inf)
    {
      if (x < 0)
	{
	  *option_lb = inf;
	  *option_ub = inf;
	}
      else if (x > 0)
	{
	  *option_lb = -inf;
	  *option_ub = -inf;
	}
      else
	{
	  *option_lb = -inf;
	  *option_ub = inf;
	}
    }
  else if (y >= inf)
    {
      if (x < 0)
	{
	  *option_lb = -inf;
	  *option_ub = -inf;
	}
      else if (x > 0)
	{
	  *option_lb = inf;
	  *option_ub = inf;
	}
      else
	{
	  *option_lb = -inf;
	  *option_ub = inf;
	}      
    }
  else
    {
      *option_lb = x * y;
      *option_ub = x * y;
    }
}


void mul(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub)
{
  double option1_lb;
  double option2_lb;
  double option3_lb;
  double option4_lb;

  double option1_ub;
  double option2_ub;
  double option3_ub;
  double option4_ub;

  _get_mul_option(xl, yl, &option1_lb, &option1_ub);
  _get_mul_option(xl, yu, &option2_lb, &option2_ub);
  _get_mul_option(xu, yl, &option3_lb, &option3_ub);
  _get_mul_option(xu, yu, &option4_lb, &option4_ub);

  double lb = option1_lb;
  if (option2_lb < lb)
    {
      lb = option2_lb;
    }
  if (option3_lb < lb)
    {
      lb = option3_lb;
    }
  if (option4_lb < lb)
    {
      lb = option4_lb;
    }

  double ub = option1_ub;
  if (option2_ub > ub)
    {
      ub = option2_ub;
    }
  if (option3_ub > ub)
    {
      ub = option3_ub;
    }
  if (option4_ub > ub)
    {
      ub = option4_ub;
    }

  *res_lb = lb;
  *res_ub = ub;
}


void inv(double xl, double xu, double* res_lb, double* res_ub, double feasibility_tol)
{
  /*
    The case where xl is very slightly positive but should be very slightly negative (or xu is very slightly negative
    but should be very slightly positive) should not be an issue. Suppose xu is 2 and xl is 1e-15 but should be -1e-15.
    The bounds obtained from this function will be [0.5, 1e15] or [0.5, inf), depending on the value of
    feasibility_tol. The true bounds are (-inf, -1e15] U [0.5, inf), where U is union. The exclusion of (-inf, -1e15]
    should be acceptable. Additionally, it very important to return a non-negative interval when xl is non-negative.
  */

  if (xu - xl <= -feasibility_tol)
    {
      throw py::value_error("lower bound is greater than upper bound in inv; xl: " + std::to_string(xl) + "; xu: " + std::to_string(xu));
    }
  else if (xu <= 0 && 0 <= xl)
    {
      // This has to return -inf to inf because it could later be multiplied by 0
      *res_lb = -inf;
      *res_ub = inf;
    }
  else if (xl < 0 && 0 < xu)
    {
      *res_lb = -inf;
      *res_ub = inf;
    }
  else if (0 <= xl && xl <= feasibility_tol)
    {
      // xu must be strictly positive
      *res_ub = inf;
      *res_lb = 1.0 / xu;
    }
  else if (xl > feasibility_tol)
    {
      // xl and xu must be strictly positive
      *res_ub = 1.0 / xl;
      *res_lb = 1.0 / xu;
    }
  else if (-feasibility_tol <= xu && xu <= 0)
    {
      // xl must be strictly negative
      *res_lb = -inf;
      *res_ub = 1.0 / xl;
    }
  else if (xu < -feasibility_tol)
    {
      // xl and xu must be strictly negative
      *res_ub = 1.0 / xl;
      *res_lb = 1.0 / xu;
    }
  else
    {
      // everything else
      *res_lb = -inf;
      *res_ub = inf;
    }
}


void div(double xl, double xu, double yl, double yu, double* res_lb, double* res_ub, double feasibility_tol)
{
  double inv_lb;
  double inv_ub;
  inv(yl, yu, &inv_lb, &inv_ub, feasibility_tol);
  mul(xl, xu, inv_lb, inv_ub, res_lb, res_ub);
}
