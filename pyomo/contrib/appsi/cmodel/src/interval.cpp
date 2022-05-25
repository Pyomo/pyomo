#include "interval.hpp"

bool _is_inf(double x) {
  if (x <= -inf || x >= inf)
    return true;
  return false;
}

void interval_add(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub) {
  if (xl <= -inf || yl <= -inf) {
    *res_lb = -inf;
  } else if (xl >= inf || yl >= inf) {
    *res_lb = inf;
  } else {
    *res_lb = xl + yl;
  }

  if (xu >= inf || yu >= inf) {
    *res_ub = inf;
  } else if (xu <= -inf || yu <= -inf) {
    *res_ub = -inf;
  } else {
    *res_ub = xu + yu;
  }
}

void interval_sub(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub) {
  if (xl <= -inf || yu >= inf) {
    *res_lb = -inf;
  } else if (xl >= inf || yu <= -inf) {
    *res_lb = inf;
  } else {
    *res_lb = xl - yu;
  }

  if (xu >= inf || yl <= -inf) {
    *res_ub = inf;
  } else if (xu <= -inf || yl >= inf) {
    *res_ub = -inf;
  } else {
    *res_ub = xu - yl;
  }
}

void _get_mul_option(double x, double y, double *option_lb, double *option_ub) {
  if (x <= -inf) {
    if (y < 0) {
      *option_lb = inf;
      *option_ub = inf;
    } else if (y > 0) {
      *option_lb = -inf;
      *option_ub = -inf;
    } else {
      *option_lb = -inf;
      *option_ub = inf;
    }
  } else if (x >= inf) {
    if (y < 0) {
      *option_lb = -inf;
      *option_ub = -inf;
    } else if (y > 0) {
      *option_lb = inf;
      *option_ub = inf;
    } else {
      *option_lb = -inf;
      *option_ub = inf;
    }
  } else if (y <= -inf) {
    if (x < 0) {
      *option_lb = inf;
      *option_ub = inf;
    } else if (x > 0) {
      *option_lb = -inf;
      *option_ub = -inf;
    } else {
      *option_lb = -inf;
      *option_ub = inf;
    }
  } else if (y >= inf) {
    if (x < 0) {
      *option_lb = -inf;
      *option_ub = -inf;
    } else if (x > 0) {
      *option_lb = inf;
      *option_ub = inf;
    } else {
      *option_lb = -inf;
      *option_ub = inf;
    }
  } else {
    *option_lb = x * y;
    *option_ub = x * y;
  }
}

void interval_mul(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub) {
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
  if (option2_lb < lb) {
    lb = option2_lb;
  }
  if (option3_lb < lb) {
    lb = option3_lb;
  }
  if (option4_lb < lb) {
    lb = option4_lb;
  }

  double ub = option1_ub;
  if (option2_ub > ub) {
    ub = option2_ub;
  }
  if (option3_ub > ub) {
    ub = option3_ub;
  }
  if (option4_ub > ub) {
    ub = option4_ub;
  }

  *res_lb = lb;
  *res_ub = ub;
}

void interval_inv(double xl, double xu, double *res_lb, double *res_ub,
                  double feasibility_tol) {
  /*
    The case where xl is very slightly positive but should be very slightly
    negative (or xu is very slightly negative but should be very slightly
    positive) should not be an issue. Suppose xu is 2 and xl is 1e-15 but should
    be -1e-15. The bounds obtained from this function will be [0.5, 1e15] or
    [0.5, inf), depending on the value of feasibility_tol. The true bounds are
    (-inf, -1e15] U [0.5, inf), where U is union. The exclusion of (-inf, -1e15]
    should be acceptable. Additionally, it very important to return a
    non-negative interval when xl is non-negative.
  */

  if (xu - xl <= -feasibility_tol) {
    throw InfeasibleConstraintException(
        "lower bound is greater than upper bound in interval_inv; xl: " +
        std::to_string(xl) + "; xu: " + std::to_string(xu));
  } else if (xu <= 0 && 0 <= xl) {
    // This has to return -inf to inf because it could later be multiplied by 0
    *res_lb = -inf;
    *res_ub = inf;
  } else if (xl < 0 && 0 < xu) {
    *res_lb = -inf;
    *res_ub = inf;
  } else if (0 <= xl && xl <= feasibility_tol) {
    // xu must be strictly positive
    *res_ub = inf;
    *res_lb = 1.0 / xu;
  } else if (xl > feasibility_tol) {
    // xl and xu must be strictly positive
    *res_ub = 1.0 / xl;
    *res_lb = 1.0 / xu;
  } else if (-feasibility_tol <= xu && xu <= 0) {
    // xl must be strictly negative
    *res_lb = -inf;
    *res_ub = 1.0 / xl;
  } else if (xu < -feasibility_tol) {
    // xl and xu must be strictly negative
    *res_ub = 1.0 / xl;
    *res_lb = 1.0 / xu;
  } else {
    // everything else
    *res_lb = -inf;
    *res_ub = inf;
  }
}

void interval_div(double xl, double xu, double yl, double yu, double *res_lb,
                  double *res_ub, double feasibility_tol) {
  double inv_lb;
  double inv_ub;
  interval_inv(yl, yu, &inv_lb, &inv_ub, feasibility_tol);
  interval_mul(xl, xu, inv_lb, inv_ub, res_lb, res_ub);
}

double _div_with_inf(double x, double y) {
  double res;
  if (_is_inf(x) && _is_inf(y))
    throw py::value_error("Cannot divide inf by inf");
  else if (y == 0)
    throw py::value_error("Division by 0.");
  else if (x <= -inf) {
    if (y < 0)
      res = inf;
    else
      res = -inf;
  } else if (x >= inf) {
    if (y < 0)
      res = -inf;
    else
      res = inf;
  } else if (_is_inf(y))
    res = 0;
  else
    res = x / y;
  return res;
}

double _log_with_inf(double x) {
  double res;
  if (x < 0)
    throw py::value_error("Cannot take the log of a negative number");
  else if (x == 0)
    res = -inf;
  else if (x >= inf)
    res = inf;
  else
    res = std::log(x);
  return res;
}

double _log10_with_inf(double x) {
  double res;
  if (x < 0)
    throw py::value_error("Cannot take the log10 of a negative number");
  else if (x == 0)
    res = -inf;
  else if (x >= inf)
    res = inf;
  else
    res = std::log10(x);
  return res;
}

double _exp_with_inf(double x) {
  double res;
  if (x <= -inf)
    res = 0;
  else if (x >= inf)
    res = inf;
  else {
    try {
      res = std::exp(x);
    } catch (const std::overflow_error &e) {
      res = inf;
    }
  }
  return res;
}

double _sin_with_inf(double x) {
  double res;
  if (_is_inf(x))
    throw py::value_error("can only compute sin of finite numbers");
  else
    res = std::sin(x);
  return res;
}

double _cos_with_inf(double x) {
  double res;
  if (_is_inf(x))
    throw py::value_error("can only compute cos of finite numbers");
  else
    res = std::cos(x);
  return res;
}

double _tan_with_inf(double x) {
  double res;
  if (_is_inf(x))
    throw py::value_error("can only compute tan of finite numbers");
  else
    res = std::tan(x);
  return res;
}

double _pow_with_inf(double x, double y) {
  double res;
  if (x == 0) {
    if (y <= -inf)
      res = inf;
    else if (y < 0)
      throw py::value_error("0 cannot be raised to a negative power");
    else if (y == 0)
      res = 1;
    else
      res = 0;
  } else if (x <= -inf) {
    if (y >= inf)
      res = inf;
    else if (y == 0)
      res = 1;
    else if (y < 0)
      res = 0;
    else if (y == round(y)) {
      int y_int = static_cast<int>(y);
      if (y_int % 2 == 0)
        res = inf;
      else
        res = -inf;
    } else
      res = inf;
  } else if (x >= inf) {
    if (y == 0)
      res = 1;
    else if (y < 0)
      res = 0;
    else
      res = inf;
  } else if (y <= -inf) {
    if (x < -1)
      res = 0;
    else if (x == -1)
      res = 1;
    else if (x < 1)
      res = inf;
    else if (x == 1)
      res = 1;
    else
      res = 0;
  } else if (y >= inf) {
    if (x < -1)
      res = inf;
    else if (x == -1)
      res = 1;
    else if (x < 1)
      res = 0;
    else if (x == 1)
      res = 1;
    else
      res = inf;
  } else if (x < 0) {
    if (y == 0)
      res = 1;
    else if (y == round(y))
      res = std::pow(x, y);
    else
      throw py::value_error(
          "cannot raise a negative number to a fractional power");
  } else
    res = std::pow(x, y);
  return res;
}

void interval_power(double xl, double xu, double yl, double yu, double *res_lb,
                    double *res_ub, double feasibility_tol) {
  /*
  Compute bounds on x**y.
  */
  if (xl > 0) {
    // If x is always positive, things are simple. We only need to worry about
    // the sign of y.
    if (yl < 0 && 0 < yu) {
      *res_lb = std::min(_pow_with_inf(xu, yl), _pow_with_inf(xl, yu));
      *res_ub = std::max(_pow_with_inf(xl, yl), _pow_with_inf(xu, yu));
    } else if (yl >= 0) {
      *res_lb = std::min(_pow_with_inf(xl, yl), _pow_with_inf(xl, yu));
      *res_ub = std::max(_pow_with_inf(xu, yl), _pow_with_inf(xu, yu));
    } else {
      *res_lb = std::min(_pow_with_inf(xu, yl), _pow_with_inf(xu, yu));
      *res_ub = std::max(_pow_with_inf(xl, yl), _pow_with_inf(xl, yu));
    }
  } else if (xl == 0) {
    if (yl >= 0) {
      *res_lb = std::min(_pow_with_inf(xl, yl), _pow_with_inf(xl, yu));
      *res_ub = std::max(_pow_with_inf(xu, yl), _pow_with_inf(xu, yu));
    } else if (yu <= 0) {
      double lb1, ub1, lb2, ub2;
      interval_sub(0, 0, yl, yu, &lb1, &ub1);
      interval_power(xl, xu, lb1, ub1, &lb2, &ub2, feasibility_tol);
      interval_inv(lb2, ub2, res_lb, res_ub, feasibility_tol);
    } else {
      double lb1, ub1, lb2, ub2;
      interval_power(xl, xu, 0, yu, &lb1, &ub1, feasibility_tol);
      interval_power(xl, xu, yl, 0, &lb2, &ub2, feasibility_tol);
      *res_lb = std::min(lb1, lb2);
      *res_ub = std::max(ub1, ub2);
    }
  } else if (yl == yu && yl == round(yl)) {
    /*
    the exponent is an integer, so x can be negative

    The logic here depends on several things:
    1) The sign of x
    2) The sign of y
    3) Whether y is even or odd.

    There are also special cases to avoid math domain errors.
     */
    int y = static_cast<int>(yl);
    if (xu <= 0) {
      if (y < 0) {
        if (y % 2 == 0) {
          *res_lb = _pow_with_inf(xl, yl);
          if (xu == 0)
            *res_ub = inf;
          else
            *res_ub = _pow_with_inf(xu, yl);
        } else {
          if (xu == 0) {
            *res_lb = -inf;
            *res_ub = inf;
          } else {
            *res_lb = _pow_with_inf(xu, yl);
            *res_ub = _pow_with_inf(xl, yl);
          }
        }
      } else {
        if (y % 2 == 0) {
          *res_lb = _pow_with_inf(xu, yl);
          *res_ub = _pow_with_inf(xl, yl);
        } else {
          *res_lb = _pow_with_inf(xl, yl);
          *res_ub = _pow_with_inf(xu, yl);
        }
      }
    } else {
      if (y < 0) {
        if (y % 2 == 0) {
          *res_lb = std::min(_pow_with_inf(xl, yl), _pow_with_inf(xu, yl));
          *res_ub = inf;
        } else {
          *res_lb = -inf;
          *res_ub = inf;
        }
      } else {
        if (y % 2 == 0) {
          *res_lb = 0;
          *res_ub = std::max(_pow_with_inf(xl, yl), _pow_with_inf(xu, yl));
        } else {
          *res_lb = _pow_with_inf(xl, yl);
          *res_ub = _pow_with_inf(xu, yl);
        }
      }
    }
  } else if (yl == yu) {
    if (xu < 0)
      throw InfeasibleConstraintException(
          "Cannot raise a negative number to a fractional power.");
    interval_power(0, xu, yl, yu, res_lb, res_ub, feasibility_tol);
  } else {
    *res_lb = -inf;
    *res_ub = inf;
  }
}

void interval_exp(double xl, double xu, double *res_lb, double *res_ub) {
  *res_lb = _exp_with_inf(xl);
  *res_ub = _exp_with_inf(xu);
}

void interval_log(double xl, double xu, double *res_lb, double *res_ub) {
  if (xl > 0)
    *res_lb = _log_with_inf(xl);
  else
    *res_lb = -inf;
  if (xu > 0)
    *res_ub = _log_with_inf(xu);
  else
    *res_ub = -inf;
}

void interval_abs(double xl, double xu, double *res_lb, double *res_ub) {
  double abs_xl = std::fabs(xl);
  double abs_xu = std::fabs(xu);
  if (xl <= 0 && xu >= 0) {
    *res_lb = 0;
    *res_ub = std::max(abs_xl, abs_xu);
  } else {
    *res_lb = std::min(abs_xl, abs_xu);
    *res_ub = std::max(abs_xl, abs_xu);
  }
}

void _inverse_abs(double zl, double zu, double *xl, double *xu) {
  if (zl < 0)
    zl = 0;
  if (zu < 0)
    zu = 0;
  *xu = std::max(zl, zu);
  *xl = -(*xu);
}

void _inverse_power1(double zl, double zu, double yl, double yu, double orig_xl,
                     double orig_xu, double *xl, double *xu,
                     double feasibility_tol) {
  /*
    z = x**y => compute bounds on x.

    First, start by computing bounds on x with

      x = exp(ln(z) / y)

    However, if y is an integer, then x can be negative, so there are several
    special cases. See the docs below.
  */
  interval_log(zl, zu, xl, xu);
  interval_div(*xl, *xu, yl, yu, xl, xu, feasibility_tol);
  interval_exp(*xl, *xu, xl, xu);

  // if y is an integer, then x can be negative
  if ((yl == yu) && (yl == round(yl))) // y is a fixed integer
  {
    int y = static_cast<int>(yl);
    if (y == 0) {
      // Anything to the power of 0 is 1, so if y is 0, then x can be anything
      // (assuming zl <= 1 <= zu, which is enforced when traversing the tree in
      // the other direction)
      *xl = -inf;
      *xu = inf;
    } else if (y % 2 == 0) {
      /*
        if y is even, then there are two primary cases (note that it is much
        easier to walk through these while looking at plots): case 1: y is
        positive x**y is convex, positive, and symmetric. The bounds on x depend
        on the lower bound of z. If zl <= 0, then xl should simply be -xu.
        However, if zl > 0, then we may be able to say something better. For
          example, if the original lower bound on x is positive, then we can
        keep xl computed from x = exp(ln(z) / y). Furthermore, if the original
        lower bound on x is larger than -xl computed from x = exp(ln(z) / y),
        then we can still keep the xl computed from x = exp(ln(z) / y). Similar
        logic applies to the upper bound of x. case 2: y is negative The ideas
        are similar to case 1.
      */
      if (zu + feasibility_tol < 0)
        throw InfeasibleConstraintException(
            "Infeasible. Anything to the power of an even integer must be "
            "positive.");
      if (y > 0) {
        double _xl, _xu;
        if (zu <= 0) {
          _xl = 0;
          _xu = 0;
        } else if (zl <= 0) {
          _xl = -(*xu);
          _xu = *xu;
        } else {
          if (orig_xl <= -(*xl) + feasibility_tol)
            _xl = -(*xu);
          else
            _xl = *xl;
          if (orig_xu < (*xl) - feasibility_tol)
            _xu = -(*xl);
          else
            _xu = *xu;
        }
        *xl = _xl;
        *xu = _xu;
      } else {
        double _xl, _xu;
        if (zu == 0)
          throw InfeasibleConstraintException(
              "Infeasible. Anything to the power of an even integer must be "
              "positive.");
        else if (zl <= 0) {
          _xl = -inf;
          _xu = inf;
        } else {
          if (orig_xl <= -(*xl) + feasibility_tol)
            _xl = -(*xu);
          else
            _xl = *xl;
          if (orig_xu < (*xl) - feasibility_tol)
            _xu = -(*xl);
          else
            _xu = *xu;
        }
        *xl = _xl;
        *xu = _xu;
      }
    } else {
      /*
        y is odd.
        Case 1: y is positive
            x**y is monotonically increasing. If y is positive, then we can can
        compute the bounds on x using x = z**(1/y) and the signs on xl and xu
        depend on the signs of zl and zu. Case 2: y is negative Again, this is
        easier to visualize with a plot. x**y approaches zero when x approaches
        -inf or inf. Thus, if zl < 0 < zu, then no bounds can be inferred for x.
        If z is positive (zl >=0 ) then we can use the bounds computed from x =
        exp(ln(z) / y). If z is negative (zu <= 0), then we live in the bottom
        left quadrant, xl depends on zu, and xu depends on zl.
      */
      if (y > 0) {
        *xl = std::abs(_pow_with_inf(std::abs(zl), _div_with_inf(1.0, yl)));
        if (zl < 0)
          *xl = -(*xl);
        *xu = std::abs(_pow_with_inf(std::abs(zu), _div_with_inf(1.0, yl)));
        if (zu < 0)
          *xu = -(*xu);
      } else {
        if (zl >= 0)
          ;
        else if (zu <= 0) {
          if (zu == 0)
            *xl = -inf;
          else
            *xl = -_pow_with_inf(std::abs(zu), _div_with_inf(1.0, yl));
          if (zl == 0)
            *xu = -inf;
          else
            *xu = -_pow_with_inf(std::abs(zl), _div_with_inf(1.0, yl));
        } else {
          *xl = -inf;
          *xu = inf;
        }
      }
    }
  }
}

void _inverse_power2(double zl, double zu, double xl, double xu, double *yl,
                     double *yu, double feasibility_tol) {
  /*
    z = x**y => compute bounds on y
    y = ln(z) / ln(x)

    This function assumes the exponent can be fractional, so x must be positive.
    This method should not be called if the exponent is an integer.
  */
  if (xu <= 0)
    throw IntervalException(
        "Cannot raise a negative variable to a fractional power.");
  if ((xl > 0 && zu <= 0) || (xl >= 0 && zu < 0))
    throw InfeasibleConstraintException("A positive variable raised to the "
                                        "power of anything must be positive.");
  double lba, uba, lbb, ubb;
  interval_log(zl, zu, &lba, &uba);
  interval_log(xl, xu, &lbb, &ubb);
  interval_div(lba, uba, lbb, ubb, yl, yu, feasibility_tol);
}

void interval_log10(double xl, double xu, double *res_lb, double *res_ub) {
  if (xl > 0)
    *res_lb = _log10_with_inf(xl);
  else
    *res_lb = -inf;
  if (xu > 0)
    *res_ub = _log10_with_inf(xu);
  else
    *res_ub = -inf;
}

void interval_sin(double xl, double xu, double *res_lb, double *res_ub) {
  // if there is a minimum between xl and xu, then the lower bound is -1.
  // Minimums occur at 2*pi*n - pi/2 find the minimum value of i such that
  // 2*pi*i - pi/2 >= xl. Then round i up. If 2*pi*i - pi/2 is still less than
  // or equal to xu, then there is a minimum between xl and xu. Thus the lb is
  // -1. Otherwise, the minimum occurs at either xl or xu
  if (xl <= -inf || xu >= inf) {
    *res_lb = -1;
    *res_ub = 1;
  } else {
    if (_is_inf(xl) || _is_inf(xu))
      throw InfeasibleConstraintException("xl is inf or xu is -inf.");

    double pi = M_PI;
    double i = (xl + pi / 2) / (2 * pi);
    i = ceil(i);
    double x_at_min = 2 * pi * i - pi / 2;
    if (x_at_min <= xu)
      *res_lb = -1;
    else
      *res_lb = std::min(_sin_with_inf(xl), _sin_with_inf(xu));

    // if there is a maximum between xl and xu, then the upper bound is 1.
    // Maximums occur at 2*pi*n + pi/2
    i = (xu - pi / 2) / (2 * pi);
    i = floor(i);
    double x_at_max = 2 * pi * i + pi / 2;
    if (x_at_max >= xl)
      *res_ub = 1;
    else
      *res_ub = std::max(_sin_with_inf(xl), _sin_with_inf(xu));
  }
}

void interval_cos(double xl, double xu, double *res_lb, double *res_ub) {
  // if there is a minimum between xl and xu, then the lower bound is -1.
  // Minimums occur at 2*pi*n - pi find the minimum value of i such that 2*pi*i
  // - pi >= xl. Then round i up. If 2*pi*i - pi/2 is still less than or equal
  // to xu, then there is a minimum between xl and xu. Thus the lb is -1.
  // Otherwise, the minimum occurs at either xl or xu
  if (xl <= -inf || xu >= inf) {
    *res_lb = -1;
    *res_ub = 1;
  } else {
    if (_is_inf(xl) || _is_inf(xu))
      throw InfeasibleConstraintException("xl is inf or xu is -inf.");

    double pi = M_PI;
    double i = (xl + pi) / (2 * pi);
    i = ceil(i);
    double x_at_min = 2 * pi * i - pi;
    if (x_at_min <= xu)
      *res_lb = -1;
    else
      *res_lb = std::min(_cos_with_inf(xl), _cos_with_inf(xu));

    // if there is a maximum between xl and xu, then the upper bound is 1.
    // Maximums occur at 2*pi*n
    i = (xu) / (2 * pi);
    i = floor(i);
    double x_at_max = 2 * pi * i;
    if (x_at_max >= xl)
      *res_ub = 1;
    else
      *res_ub = std::max(_cos_with_inf(xl), _cos_with_inf(xu));
  }
}

void interval_tan(double xl, double xu, double *res_lb, double *res_ub) {
  // tan goes to -inf and inf at every pi*i + pi/2 (integer i). If one of these
  // values is between xl and xu, then the lb is -inf and the ub is inf.
  // Otherwise the minimum occurs at xl and the maximum occurs at xu. find the
  // minimum value of i such that pi*i + pi/2 >= xl. Then round i up. If pi*i +
  // pi/2 is still less than or equal to xu, then there is an undefined point
  // between xl and xu.
  if ((xl <= -inf) || (xu >= inf)) {
    *res_lb = -inf;
    *res_ub = inf;
  } else if (_is_inf(xl) || _is_inf(xu))
    throw InfeasibleConstraintException("xl is inf or xu is -inf.");
  else {
    double pi = M_PI;
    double i = (xl - pi / 2) / (pi);
    i = ceil(i);
    double x_at_undefined = pi * i + pi / 2;
    if (x_at_undefined <= xu) {
      *res_lb = -inf;
      *res_ub = inf;
    } else {
      *res_lb = _tan_with_inf(xl);
      *res_ub = _tan_with_inf(xu);
    }
  }
}

void interval_asin(double xl, double xu, double yl, double yu, double *res_lb,
                   double *res_ub, double feasibility_tol) {
  /*
    y = asin(x); propagate bounds from x to y
    x = sin(y)
  */
  if (xl < -1)
    xl = -1;
  if (xu > 1)
    xu = 1;

  double pi = M_PI;

  if (yl <= -inf)
    *res_lb = yl;
  else if (xl <= _sin_with_inf(yl) && _sin_with_inf(yl) <= xu)
    // if sin(yl) >= xl then yl satisfies the bounds on x, and the lower bound
    // of y cannot be improved
    *res_lb = yl;
  else if (_sin_with_inf(yl) < xl) {
    /*
      we can only push yl up from its current value to the next lowest value
      such that xl = sin(y). In other words, we need to

      min y
      s.t.
          xl = sin(y)
          y >= yl

      globally.
    */
    // first find the next minimum of x = sin(y). Minimums occur at y = 2*pi*n -
    // pi/2 for integer n.
    double i = (yl + pi / 2) / (2 * pi);
    double i1 = floor(i);
    double i2 = ceil(i);
    i1 = 2 * pi * i1 - pi / 2;
    i2 = 2 * pi * i2 - pi / 2;
    // now find the next value of y such that xl = sin(y). This can be computed
    // by a distance from the minimum (i).
    double y_tmp =
        std::asin(xl); // this will give me a value between -pi/2 and pi/2
    double dist = y_tmp - (-pi / 2); // this is the distance between the minimum
                                     // of the sin function and a value that
    // satisfies xl = sin(y)
    double lb1 = i1 + dist;
    double lb2 = i2 + dist;
    if (lb1 >= yl - feasibility_tol)
      *res_lb = lb1;
    else
      *res_lb = lb2;
  } else {
    // sin(yl) > xu
    double i = (yl - pi / 2) / (2 * pi);
    double i1 = floor(i);
    double i2 = ceil(i);
    i1 = 2 * pi * i1 + pi / 2;
    i2 = 2 * pi * i2 + pi / 2;
    double y_tmp = std::asin(xu);
    double dist = pi / 2 - y_tmp;
    double lb1 = i1 + dist;
    double lb2 = i2 + dist;
    if (lb1 >= yl - feasibility_tol)
      *res_lb = lb1;
    else
      *res_lb = lb2;
  }

  // use the same logic for the maximum
  if (yu >= inf)
    *res_ub = yu;
  else if (xl <= _sin_with_inf(yu) && _sin_with_inf(yu) <= xu)
    *res_ub = yu;
  else if (_sin_with_inf(yu) > xu) {
    double i = (yu - pi / 2) / (2 * pi);
    double i1 = ceil(i);
    double i2 = floor(i);
    i1 = 2 * pi * i1 + pi / 2;
    i2 = 2 * pi * i2 + pi / 2;
    double y_tmp = std::asin(xu);
    double dist = pi / 2 - y_tmp;
    double ub1 = i1 - dist;
    double ub2 = i2 - dist;
    if (ub1 <= yu + feasibility_tol)
      *res_ub = ub1;
    else
      *res_ub = ub2;
  } else {
    // math.sin(yu) < xl
    double i = (yu + pi / 2) / (2 * pi);
    double i1 = ceil(i);
    double i2 = floor(i);
    i1 = 2 * pi * i1 - pi / 2;
    i2 = 2 * pi * i2 - pi / 2;
    double y_tmp = std::asin(xl);
    double dist = y_tmp - (-pi / 2);
    double ub1 = i1 - dist;
    double ub2 = i2 - dist;
    if (ub1 <= yu + feasibility_tol)
      *res_ub = ub1;
    else
      *res_ub = ub2;
  }
}

void interval_acos(double xl, double xu, double yl, double yu, double *res_lb,
                   double *res_ub, double feasibility_tol) {
  /*
    y = acos(x); propagate bounds from x to y
    x = cos(y)
  */
  if (xl < -1)
    xl = -1;
  if (xu > 1)
    xu = 1;

  double pi = M_PI;

  if (yl <= -inf)
    *res_lb = yl;
  else if (xl <= _cos_with_inf(yl) && _cos_with_inf(yl) <= xu)
    // if xl <= cos(yl) <= xu then yl satisfies the bounds on x, and the lower
    // bound of y cannot be improved
    *res_lb = yl;
  else if (_cos_with_inf(yl) < xl) {
    /*
      we can only push yl up from its current value to the next lowest value
      such that xl = cos(y). In other words, we need to

      min y
      s.t.
          xl = cos(y)
          y >= yl

      globally.
    */
    // first find the next minimum of x = cos(y). Minimums occur at y = 2*pi*n -
    // pi for integer n.
    double i = (yl + pi) / (2 * pi);
    double i1 = floor(i);
    double i2 = ceil(i);
    i1 = 2 * pi * i1 - pi;
    i2 = 2 * pi * i2 - pi;
    // now find the next value of y such that xl = cos(y). This can be computed
    // by a distance from the minimum (i).
    double y_tmp = std::acos(xl); // this will give me a value between 0 and pi
    double dist = pi - y_tmp; // this is the distance between the minimum of the
                              // sin function and a value that
    // satisfies xl = sin(y)
    double lb1 = i1 + dist;
    double lb2 = i2 + dist;
    if (lb1 >= yl - feasibility_tol)
      *res_lb = lb1;
    else
      *res_lb = lb2;
  } else {
    // cos(yl) > xu
    // first find the next maximum of x = cos(y).
    double i = yl / (2 * pi);
    double i1 = floor(i);
    double i2 = ceil(i);
    i1 = 2 * pi * i1;
    i2 = 2 * pi * i2;
    double y_tmp = std::acos(xu);
    double dist = y_tmp;
    double lb1 = i1 + dist;
    double lb2 = i2 + dist;
    if (lb1 >= yl - feasibility_tol)
      *res_lb = lb1;
    else
      *res_lb = lb2;
  }

  // use the same logic for the maximum
  if (yu >= inf)
    *res_ub = yu;
  else if (xl <= _cos_with_inf(yu) && _cos_with_inf(yu) <= xu)
    *res_ub = yu;
  else if (_cos_with_inf(yu) > xu) {
    double i = yu / (2 * pi);
    double i1 = ceil(i);
    double i2 = floor(i);
    i1 = 2 * pi * i1;
    i2 = 2 * pi * i2;
    double y_tmp = std::acos(xu);
    double dist = y_tmp;
    double ub1 = i1 - dist;
    double ub2 = i2 - dist;
    if (ub1 <= yu + feasibility_tol)
      *res_ub = ub1;
    else
      *res_ub = ub2;
  } else {
    // math.cos(yu) < xl
    double i = (yu + pi) / (2 * pi);
    double i1 = ceil(i);
    double i2 = floor(i);
    i1 = 2 * pi * i1 - pi;
    i2 = 2 * pi * i2 - pi;
    double y_tmp = std::acos(xl);
    double dist = pi - y_tmp;
    double ub1 = i1 - dist;
    double ub2 = i2 - dist;
    if (ub1 <= yu + feasibility_tol)
      *res_ub = ub1;
    else
      *res_ub = ub2;
  }
}

void interval_atan(double xl, double xu, double yl, double yu, double *res_lb,
                   double *res_ub) {
  /*
    y = atan(x); propagate bounds from x to y
    x = tan(y)
  */

  double pi = M_PI;

  // tan goes to -inf and inf at every pi*i + pi/2 (integer i).
  if (xl <= -inf || yl <= -inf)
    *res_lb = yl;
  else {
    double i = (yl - pi / 2) / pi;
    i = floor(i);
    i = pi * i + pi / 2;
    double y_tmp = std::atan(xl);
    double dist = y_tmp - (-pi / 2);
    *res_lb = i + dist;
  }

  if (xu >= inf || yu >= inf)
    *res_ub = yu;
  else {
    double i = (yu - pi / 2) / pi;
    i = ceil(i);
    i = pi * i + pi / 2;
    double y_tmp = std::atan(xu);
    double dist = pi / 2 - y_tmp;
    *res_ub = i - dist;
  }

  if (yl > *res_lb)
    *res_lb = yl;
  if (yu < *res_ub)
    *res_ub = yu;
}

std::pair<double, double> py_interval_add(double xl, double xu, double yl,
                                          double yu) {
  double res_lb, res_ub;
  interval_add(xl, xu, yl, yu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_sub(double xl, double xu, double yl,
                                          double yu) {
  double res_lb, res_ub;
  interval_sub(xl, xu, yl, yu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_mul(double xl, double xu, double yl,
                                          double yu) {
  double res_lb, res_ub;
  interval_mul(xl, xu, yl, yu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_inv(double xl, double xu,
                                          double feasibility_tol) {
  double res_lb, res_ub;
  interval_inv(xl, xu, &res_lb, &res_ub, feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_div(double xl, double xu, double yl,
                                          double yu, double feasibility_tol) {
  double res_lb, res_ub;
  interval_div(xl, xu, yl, yu, &res_lb, &res_ub, feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_power(double xl, double xu, double yl,
                                            double yu, double feasibility_tol) {
  double res_lb, res_ub;
  interval_power(xl, xu, yl, yu, &res_lb, &res_ub, feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_exp(double xl, double xu) {
  double res_lb, res_ub;
  interval_exp(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_log(double xl, double xu) {
  double res_lb, res_ub;
  interval_log(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_abs(double xl, double xu) {
  double res_lb, res_ub;
  interval_abs(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> _py_inverse_abs(double zl, double zu) {
  double xl, xu;
  _inverse_abs(zl, zu, &xl, &xu);
  return std::make_pair(xl, xu);
}

std::pair<double, double> _py_inverse_power1(double zl, double zu, double yl,
                                             double yu, double orig_xl,
                                             double orig_xu,
                                             double feasibility_tol) {
  double res_lb, res_ub;
  _inverse_power1(zl, zu, yl, yu, orig_xl, orig_xu, &res_lb, &res_ub,
                  feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> _py_inverse_power2(double zl, double zu, double xl,
                                             double xu,
                                             double feasibility_tol) {
  double res_lb, res_ub;
  _inverse_power2(zl, zu, xl, xu, &res_lb, &res_ub, feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_log10(double xl, double xu) {
  double res_lb, res_ub;
  interval_log10(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_sin(double xl, double xu) {
  double res_lb, res_ub;
  interval_sin(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_cos(double xl, double xu) {
  double res_lb, res_ub;
  interval_cos(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_tan(double xl, double xu) {
  double res_lb, res_ub;
  interval_tan(xl, xu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_asin(double xl, double xu, double yl,
                                           double yu, double feasibility_tol) {
  double res_lb, res_ub;
  interval_asin(xl, xu, yl, yu, &res_lb, &res_ub, feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_acos(double xl, double xu, double yl,
                                           double yu, double feasibility_tol) {
  double res_lb, res_ub;
  interval_acos(xl, xu, yl, yu, &res_lb, &res_ub, feasibility_tol);
  return std::make_pair(res_lb, res_ub);
}

std::pair<double, double> py_interval_atan(double xl, double xu, double yl,
                                           double yu) {
  double res_lb, res_ub;
  interval_atan(xl, xu, yl, yu, &res_lb, &res_ub);
  return std::make_pair(res_lb, res_ub);
}
