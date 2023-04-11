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

from pyomo.common.deprecation import RenamedClass
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.indexed_component import rule_wrapper
from pyomo.core.base.expression import (
    Expression,
    _GeneralExpressionData,
    ScalarExpression,
    IndexedExpression,
)
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error

__all__ = ('Integral',)


@ModelComponentFactory.register("Integral Expression in a DAE model.")
class Integral(Expression):
    """
    Represents an integral over a continuous domain

    The :py:class:`Integral<pyomo.dae.Integral>` component can be used to
    represent an integral taken over the entire domain of a
    :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>`. Once every
    :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` in a model has been
    discretized, any integrals in the model will be converted to algebraic
    equations using the trapezoid rule. Future development will include more
    sophisticated numerical integration methods.

    Parameters
    ----------
    *args
        Every indexing set needed to evaluate the integral expression

    wrt : :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>`
        The continuous domain over which the integral is being taken

    rule : function
        Function returning the expression being integrated
    """

    def __new__(cls, *args, **kwds):
        if cls != Integral:
            return super(Integral, cls).__new__(cls)
        if len(args) == 0:
            raise ValueError("Integral must be indexed by a ContinuousSet")
        elif len(args) == 1:
            return ScalarIntegral.__new__(ScalarIntegral)
        else:
            return IndexedIntegral.__new__(IndexedIntegral)

    def __init__(self, *args, **kwds):
        if "wrt" in kwds and "withrespectto" in kwds:
            raise TypeError("Cannot specify both 'wrt' and 'withrespectto keywords")

        wrt = kwds.pop('wrt', None)
        wrt = kwds.pop('withrespectto', wrt)

        if wrt is None:
            # Check to be sure Integral is indexed by single
            # ContinuousSet and take Integral with respect to that
            # ContinuousSet
            if len(args) != 1:
                raise ValueError(
                    "Integral indexed by multiple ContinuousSets. "
                    "The desired ContinuousSet must be specified using the "
                    "keyword argument 'wrt'"
                )
            wrt = args[0]

        if type(wrt) is not ContinuousSet:
            raise ValueError(
                "Cannot take the integral with respect to '%s'. Must take an "
                "integral with respect to a ContinuousSet" % wrt
            )
        self._wrt = wrt

        loc = None
        for i, s in enumerate(args):
            if s is wrt:
                loc = i

        # Check that the wrt ContinuousSet is in the argument list
        if loc is None:
            raise ValueError(
                "The ContinuousSet '%s' was not found in the indexing sets "
                "of the Integral" % wrt.name
            )
        self.loc = loc

        # Remove the index that the integral is being expanded over
        arg = args[0:loc] + args[loc + 1 :]

        # Check that if bounds are given
        bounds = kwds.pop('bounds', None)
        if bounds is not None:
            raise DAE_Error(
                "Setting bounds on integrals has not yet been implemented. "
                "Integrals may only be taken over an entire ContinuousSet"
            )

        # Create integral expression and pass to the expression initialization
        intexp = kwds.pop('expr', None)
        intexp = kwds.pop('rule', intexp)
        if intexp is None:
            raise ValueError("Must specify an integral expression")

        _is_indexed = bool(len(arg))

        def _trap_rule(rule, m, *a):
            ds = sorted(m.find_component(wrt.local_name))
            return sum(
                0.5
                * (ds[i + 1] - ds[i])
                * (
                    rule(m, *(a[0:loc] + (ds[i + 1],) + a[loc:]))
                    + rule(m, *(a[0:loc] + (ds[i],) + a[loc:]))
                )
                for i in range(len(ds) - 1)
            )

        # Note that position_map is mapping arguments (block, *args), so
        # must be 1 more than len(args), and loc has to be offset by one
        kwds['rule'] = rule_wrapper(
            intexp,
            _trap_rule,
            positional_arg_map=(i for i in range(len(args) + 1) if i != loc + 1),
        )
        kwds.setdefault('ctype', Integral)
        Expression.__init__(self, *arg, **kwds)

    def get_continuousset(self):
        """Return the :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>`
        the integral is being taken over
        """
        return self._wrt


class ScalarIntegral(ScalarExpression, Integral):
    """
    An integral that will have no indexing sets after applying a numerical
    integration transformation
    """

    def __init__(self, *args, **kwds):
        _GeneralExpressionData.__init__(self, None, component=self)
        Integral.__init__(self, *args, **kwds)

    def clear(self):
        self._data = {}

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been
        discretized
        """
        if 'scheme' not in self._wrt.get_discretization_info():
            return False
        return True


class SimpleIntegral(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarIntegral
    __renamed__version__ = '6.0'


class IndexedIntegral(IndexedExpression, Integral):
    """
    An integral that will be indexed after applying a numerical integration
    transformation
    """

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been
        discretized.
        """
        wrt = self._wrt
        if 'scheme' not in wrt.get_discretization_info():
            return False

        setlist = []
        if self.dim() == 1:
            setlist = [self.index_set()]
        else:
            setlist = self.index_set().set_tuple

        for i in setlist:
            if i.ctype is ContinuousSet:
                if 'scheme' not in i.get_discretization_info():
                    return False
        return True
