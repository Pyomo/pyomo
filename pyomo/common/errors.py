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

import inspect
import textwrap


def format_exception(msg, prolog=None, epilog=None, exception=None, width=76):
    fields = []

    if epilog:
        indent = ' ' * 8
    else:
        indent = ' ' * 4

    if exception is None:
        # default to the length of 'NotImplementedError: ', the longest
        # built-in name that we commonly raise
        initial_indent = ' ' * 21
    else:
        if not inspect.isclass(exception):
            exception = exception.__class__
        initial_indent = ' ' * (len(exception.__name__) + 2)
        if exception.__module__ != 'builtins':
            initial_indent += ' ' * (len(exception.__module__) + 1)

    if prolog is not None:
        if '\n' not in prolog:
            # We want to strip off the leading indent that we added as a
            # placeholder for the string representation of the exception
            # class name.
            prolog = textwrap.fill(
                prolog,
                width=width,
                initial_indent=initial_indent,
                subsequent_indent=' ' * 4,
                break_long_words=False,
                break_on_hyphens=False,
            ).lstrip()
        # If the prolog line-wrapped, ensure that the message is
        # indented an additional level.
        if '\n' in prolog:
            indent = ' ' * 8
        fields.append(prolog)
        initial_indent = indent

    if '\n' not in msg:
        msg = textwrap.fill(
            msg,
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not fields:
            # We want to strip off the leading indent that we just
            # added, but only if there is no prolog
            msg = msg.lstrip()
    fields.append(msg)

    if epilog is not None:
        if '\n' not in epilog:
            epilog = textwrap.fill(
                epilog,
                width=width,
                initial_indent=' ' * 4,
                subsequent_indent=' ' * 4,
                break_long_words=False,
                break_on_hyphens=False,
            )
        fields.append(epilog)

    return '\n'.join(fields)


class ApplicationError(Exception):
    """
    An exception used when an external application generates an error.
    """

    pass


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

    def __str__(self):
        return format_exception(
            repr(super().__str__()),
            prolog="Internal Pyomo implementation error:",
            epilog="Please report this to the Pyomo Developers.",
            exception=self,
        )


class InvalidValueError(PyomoException, ValueError):
    """
    Exception class used for for value errors in compiled model representations
    """

    pass


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

    def __str__(self):
        return format_exception(
            repr(super().__str__()),
            prolog="Sorry, mouse, no cookies here!",
            epilog="This is functionality we think may be rational to "
            "support, but is not yet implemented (possibly due to developer "
            "availability, complexity of edge cases, or general practicality "
            "or tractability). However, please feed the mice: "
            "pull requests are always welcome!",
            exception=self,
        )
