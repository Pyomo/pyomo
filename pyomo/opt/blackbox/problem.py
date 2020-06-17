#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Utilities to support the definition of optimization applications that
can be optimized with the Acro COLIN optimizers.
"""

__all__ = ['OptProblem', 'RealOptProblem', 'MixedIntOptProblem', 'response_enum']

import os
import sys

import enum

from pyomo.opt.blackbox.problem_io import BlackBoxOptProblemIOFactory
from pyomo.opt.blackbox.point import MixedIntVars, RealVars

class response_enum(str, enum.Enum):
    FunctionValue='FunctionValue'
    FunctionValues='FunctionValues'
    Gradient='Gradient'
    Hessian='Hessian'
    NonlinearConstraintValues='NonlinearConstraintValues'
    Jacobian='Jacobian'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


class OptProblem(object):
    """
    A class that defines an application that can be optimized
    by a COLIN optimizer via system calls.
    """

    def __init__(self):
        """
        The constructor.  Derived classes should define the response types.

        By default, only function evaluations are supported in an OptProblem
        instance.
        """
        self.response_types = [response_enum.FunctionValue]


    def main(self, argv, format='colin'):
        """
        The main routine for parsing the command-line and executing
        the evaluation.
        """
        if len(argv) < 3:                               #pragma:nocover
            print(argv[0 ]+ " <input> <output> <log>")
            sys.exit(1)
        #
        # Get enum strings
        #
        self.response_str = list(map(str, self.response_types))
        #
        # Parse XML input file
        #
        iomngr = BlackBoxOptProblemIOFactory(format)
        if iomngr is None:
            raise ValueError("Unknown IO format '%s' for COLIN OptProblem" % str(format))
        if not os.path.exists(argv[1]):
            raise IOError("Unknown input file '%s'" % argv[1])
        self._compute_prefix(argv[1])
        point = self.create_point()
        point, requests = iomngr.read(argv[1], point)
        self.validate(point)
        response = self._compute_results(point, requests)
        iomngr.write(argv[2], response)

    def create_point(self):
        """
        Create the point type for this domain.
        This method is over-written to customized an OptProblem
        for the search domain.
        """
        return None                                     #pragma:nocover

    def function_value(self, point):
        """
        Compute a function value.
        """
        return None                                     #pragma:nocover

    def function_values(self, point):                   #pragma:nocover
        """
        Compute a list of function values.
        """
        val = self.function_value(point)
        if val is None:
            return []
        else:
            return [val]

    def gradient(self, point):
        """
        Compute a function gradient.
        """
        return []                                       #pragma:nocover

    def hessian(self, point):
        """
        Compute a function Hessian matrix.
        """
        return {}                                       #pragma:nocover

    def nonlinear_constraint_values(self, point):
        """
        Compute nonlinear constraint values.
        """
        return []                                       #pragma:nocover

    def jacobian(self, point):
        """
        Compute the Jacobian.
        """
        return {}                                       #pragma:nocover

    def _compute_results(self, point, requests):
        """
        Compute the requested results.
        """
        response = {}
        for key in requests:
            if key not in self.response_str:
                response[key] = "ERROR: Unsupported application request %s" % str(key)
            #
            elif key == "FunctionValue":
                response[key] = self.function_value(point)
            elif key == "FunctionValues":
                response[key] = self.function_values(point)
            elif key == "Gradient":
                response[key] = self.gradient(point)
            elif key == "NonlinearConstraintValues":
                response[key] = self.nonlinear_constraint_values(point)
            elif key == "Jacobian":
                response[key] = self.jacobian(point)
            elif key == "Hessian":
                response[key] = self.hessian(point)
            #
        return response

    def _compute_prefix(self, filename):
        base, ext = os.path.splitext(filename)
        self.prefix = base

    def validate(self, point):                          #pragma:nocover
        """
        This function should throw an exception if an error occurs
        """
        pass



class MixedIntOptProblem(OptProblem):

    def __init__(self):
        OptProblem.__init__(self)
        self.int_lower=[]
        self.int_upper=[]
        self.real_lower=[]
        self.real_upper=[]
        self.nreal=0
        self.nint=0
        self.nbinary=0

    def create_point(self):
        return MixedIntVars( nreal=self.nreal, nint=self.nint, nbinary=self.nbinary )

    def validate(self, point):
        if len(point.reals) !=  self.nreal:
            raise ValueError("Number of reals is "+str(len(point.reals))+" but this problem is configured for "+str(self.nreal))
        if len(point.ints) !=  self.nint:
            raise ValueError("Number of integers is "+str(len(point.ints))+" but this problem is configured for "+str(self.nint))
        if len(point.bits) !=  self.nbinary:
            raise ValueError("Number of binaries is "+str(len(point.bits))+" but this problem is configured for "+str(self.nbinary))
        if len(self.int_lower) > 0:
            for i in range(0,self.nint):
                if self.int_lower[i] is not None and self.int_lower[i] > point.ints[i]:
                    raise ValueError("Integer "+str(i)+" has a value "+str(point.ints[i])+" that is lower than the integer lower bound "+str(self.int_lower[i]))
                if self.int_upper[i] is not None and self.int_upper[i] < point.ints[i]:
                    raise ValueError("Integer "+str(i)+" has a value "+str(point.ints[i])+" that is higher than the integer upper bound "+str(self.int_upper[i]))

        if len(self.real_lower) > 0:
            for i in range(0,self.nreal):
                if self.real_lower[i] is not None and self.real_lower[i] > point.reals[i]:
                    raise ValueError("Real "+str(i)+" has a value "+str(point.reals[i])+" that is lower than the real lower bound "+str(self.real_lower[i]))
                if self.real_upper[i] is not None and self.real_upper[i] < point.reals[i]:
                    raise ValueError("Real "+str(i)+" has a value "+str(point.reals[i])+" that is higher than the real upper bound "+str(self.real_upper[i]))



class RealOptProblem(OptProblem):

    def __init__(self):
        OptProblem.__init__(self)
        self.lower=[]
        self.upper=[]
        self.nvars=0

    def create_point(self):
        """
        Create the point type for this domain.
        """
        return RealVars( nreal=self.nvars )

    def validate(self, point):
        if len(point.vars) !=  self.nvars:
            raise ValueError("Number of real variables is "+str(len(point.vars))+" but this problem is configured for "+str(self.nvars))
        if len(self.lower) > 0:
            for i in range(0,self.nvars):
                if self.lower[i] is not None and self.lower[i] > point.vars[i]:
                    raise ValueError("Variable "+str(i)+" has a value "+str(point.vars[i])+" that is lower than the real lower bound "+str(self.lower[i]))
                if self.upper[i] is not None and self.upper[i] < point.vars[i]:
                    raise ValueError("Variable "+str(i)+" has a value "+str(point.vars[i])+" that is higher than the real upper bound "+str(self.upper[i]))


