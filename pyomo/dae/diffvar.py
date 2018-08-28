#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import weakref
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.dae.contset import ContinuousSet
from six import iterkeys

__all__ = ('DerivativeVar', 'DAE_Error',)


def create_access_function(var):
    """
    This method returns a function that returns a component by calling
    it rather than indexing it
    """
    def _fun(*args):
        return var[args]
    return _fun


class DAE_Error(Exception):
    """Exception raised while processing DAE Models"""


@ModelComponentFactory.register("Derivative of a Var in a DAE model.")
class DerivativeVar(Var):
    """
    Represents derivatives in a model and defines how a
    :py:class:`Var<pyomo.environ.Var>` is differentiated

    The :py:class:`DerivativeVar <pyomo.dae.DerivativeVar>` component is
    used to declare a derivative of a :py:class:`Var <pyomo.environ.Var>`.
    The constructor accepts a single positional argument which is the
    :py:class:`Var<pyomo.environ.Var>` that's being differentiated. A
    :py:class:`Var <pyomo.environ.Var>` may only be differentiated with
    respect to a :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` that it
    is indexed by. The indexing sets of a :py:class:`DerivativeVar
    <pyomo.dae.DerivativeVar>` are identical to those of the :py:class:`Var
    <pyomo.environ.Var>` it is differentiating.

    Parameters
    ----------
    sVar : ``pyomo.environ.Var``
        The variable being differentiated

    wrt : ``pyomo.dae.ContinuousSet`` or tuple
        Equivalent to `withrespectto` keyword argument. The
        :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` that the
        derivative is being taken with respect to. Higher order derivatives
        are represented by including the
        :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` multiple times in
        the tuple sent to this keyword. i.e. ``wrt=(m.t, m.t)`` would be the
        second order derivative with respect to ``m.t``
    """

    # Private Attributes:
    # _stateVar   The :class:`Var` being differentiated
    # _wrt        A list of the :class:`ContinuousSet` components the
    #             derivative is being taken with respect to
    # _expr       An expression representing the discretization equations
    #             linking the :class:`DerivativeVar` to its state :class:`Var`.

    def __init__(self, sVar, **kwds):

        if not isinstance(sVar, Var):
            raise DAE_Error(
                "%s is not a variable. Can only take the derivative of a Var"
                "component." % sVar)

        if "wrt" in kwds and "withrespectto" in kwds:
            raise TypeError(
                "Cannot specify both 'wrt' and 'withrespectto keywords "
                "in a DerivativeVar")

        wrt = kwds.pop('wrt', None)
        wrt = kwds.pop('withrespectto', wrt)

        try:
            num_contset = len(sVar._contset)
        except:
            sVar._contset = {}
            sVar._derivative = {}
            if sVar.dim() == 0:
                num_contset = 0
            elif sVar.dim() == 1:
                sidx_sets = sVar._index
                if sidx_sets.type() is ContinuousSet:
                    sVar._contset[sidx_sets] = 0
            else:
                sidx_sets = sVar._implicit_subsets
                for i, s in enumerate(sidx_sets):
                    if s.type() is ContinuousSet:
                        sVar._contset[s] = i
            num_contset = len(sVar._contset)

        if num_contset == 0:
            raise DAE_Error(
                "The variable %s is not indexed by any ContinuousSets. A "
                "derivative may only be taken with respect to a continuous "
                "domain" % sVar)

        if wrt is None:
            # Check to be sure Var is indexed by single ContinuousSet and take
            # first deriv wrt that set
            if num_contset != 1:
                raise DAE_Error(
                    "The variable %s is indexed by multiple ContinuousSets. "
                    "The desired ContinuousSet must be specified using the "
                    "keyword argument 'wrt'" % sVar)
            wrt = [next(iterkeys(sVar._contset)), ]
        elif type(wrt) is ContinuousSet:
            if wrt not in sVar._contset:
                raise DAE_Error(
                    "Invalid derivative: The variable %s is not indexed by "
                    "the ContinuousSet %s" % (sVar, wrt))
            wrt = [wrt, ]
        elif type(wrt) is tuple or type(wrt) is list:
            for i in wrt:
                if type(i) is not ContinuousSet:
                    raise DAE_Error(
                        "Cannot take the derivative with respect to %s. "
                        "Expected a ContinuousSet or a tuple of "
                        "ContinuousSets" % i)
                if i not in sVar._contset:
                    raise DAE_Error(
                        "Invalid derivative: The variable %s is not indexed "
                        "by the ContinuousSet %s" % (sVar, i))
            wrt = list(wrt)
        else:
            raise DAE_Error(
                "Cannot take the derivative with respect to %s. "
                "Expected a ContinuousSet or a tuple of ContinuousSets" % i)

        wrtkey = [str(i) for i in wrt]
        wrtkey.sort()
        wrtkey = tuple(wrtkey)

        if wrtkey in sVar._derivative:
            raise DAE_Error(
                "Cannot create a new derivative variable for variable "
                "%s: derivative already defined as %s"
                % (sVar.name, sVar._derivative[wrtkey]().name))

        sVar._derivative[wrtkey] = weakref.ref(self)
        self._sVar = sVar
        self._wrt = wrt

        kwds.setdefault('ctype', DerivativeVar)

        if sVar._implicit_subsets is None:
            arg = (sVar.index_set(),)
        else:
            arg = tuple(sVar._implicit_subsets)

        Var.__init__(self,*arg,**kwds)

    def get_continuousset_list(self):
        """ Return the a list of :py:class:`ContinuousSet` components the
        derivative is being taken with respect to.

        Returns
        -------
        `list`
        """
        return self._wrt

    def is_fully_discretized(self):
        """
        Check to see if all the
        :py:class:`ContinuousSets<pyomo.dae.ContinuousSet>` this derivative
        is taken with respect to have been discretized.

        Returns
        -------
        `boolean`
        """
        for i in self._wrt:
            if 'scheme' not in i.get_discretization_info():
                return False
        return True

    def get_state_var(self):
        """ Return the :py:class:`Var` that is being differentiated.

        Returns
        -------
        :py:class:`Var<pyomo.environ.Var>`
        """
        return self._sVar

    def get_derivative_expression(self):
        """
        Returns the current discretization expression for this derivative or
        creates an access function to its :py:class:`Var` the first time
        this method is called. The expression gets built up as the
        discretization transformations are sequentially applied to each
        :py:class:`ContinuousSet` in the model.
        """
        try:
            return self._expr
        except:
            self._expr = create_access_function(self._sVar)
            return self._expr

    def set_derivative_expression(self, expr):
        """ Sets``_expr``, an expression representing the discretization
        equations linking the :class:`DerivativeVar` to its state
        :class:`Var`
        """
        self._expr = expr

