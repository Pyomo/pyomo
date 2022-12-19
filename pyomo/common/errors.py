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


class ApplicationError(Exception):
    """
    An exception used when an external application generates an error.
    """

    def __init__(self, *args, **kargs):
        Exception.__init__(self, *args, **kargs)  #pragma:nocover


class PyomoException(Exception):
    """
    Exception class for other pyomo exceptions to inherit from,
    allowing pyomo exceptions to be caught in a general way
    (e.g., in other applications that use Pyomo).
    """
    pass


class DeveloperError(PyomoException, NotImplementedError):
    """
    Exception class used to throw errors that result from Pyomo
    programming errors, rather than user modeling errors (e.g., a
    component not declaring a 'ctype').
    """

    def __init__(self, val):
        self.parameter = val

    def __str__(self):
        return ( "Internal Pyomo implementation error:\n\t%s\n"
                 "\tPlease report this to the Pyomo Developers."
                 % ( repr(self.parameter), ) )


class IntervalException(PyomoException, ValueError):
    """
    Exception class used for errors in interval arithmetic.
    """
    pass


class InfeasibleConstraintException(PyomoException):
    """
    Exception class used by Pyomo transformations to indicate
    that an infeasible constraint has been identified (e.g. in
    the course of range reduction).
    """
    pass


class NondifferentiableError(PyomoException, ValueError):
    """A Pyomo-specific ValueError raised for non-differentiable expressions"""
    pass

class TempfileContextError(PyomoException, IndexError):
    """A Pyomo-specific IndexError raised when attempting to use the
    TempfileManager when it does not have a currently active context.

    """
    pass


class MouseTrap(PyomoException, NotImplementedError):
    """
    Exception class used to throw errors for not-implemented functionality
    that might be rational to support (i.e., we already gave you a cookie)
    but risks taking Pyomo's flexibility a step beyond what is sane,
    or solvable, or communicable to a solver, etc. (i.e., Really? Now you
    want a glass of milk too?)
    """
    def __init__(self, val):
        self.parameter = val

    def __str__(self):
        return ("Sorry, mouse, no cookies here!\n\t%s\n"
                "\tThis is functionality we think may be rational to "
                "support, but is not yet implemented since it might go "
                "beyond what can practically be solved. However, please "
                "feed the mice: pull requests are always welcome!"
                % (repr(self.parameter), ))
