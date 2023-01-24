#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numeric_expr import NumericExpression

class PiecewiseLinearExpression(NumericExpression):
    def nargs(self):
        return len(self._args_)

    @property
    def _parent_pw_linear_function(self):
        return self._args_[0]

    def _to_string(self, values, verbose, smap):
        return "%s(%s)" % (values[0], ', '.join(values[1:])) 

    def polynomial_degree(self):
        return None
