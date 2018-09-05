#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from pyomo.common.timing import ConstructionTimer
from pyomo.core import *
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.sets import OrderedSimpleSet
from pyomo.core.base.numvalue import native_numeric_types

logger = logging.getLogger('pyomo.dae')
__all__ = ['ContinuousSet']


@ModelComponentFactory.register(
                   "A bounded continuous numerical range optionally containing"
                   " discrete points of interest.")
class ContinuousSet(OrderedSimpleSet):
    """ Represents a bounded continuous domain

        Minimally, this set must contain two numeric values defining the
        bounds of a continuous range. Discrete points of interest may
        be added to the continuous set. A continuous set is one
        dimensional and may only contain numerical values.

        Parameters
        ----------
        initialize : `list`
            Default discretization points to be included

        bounds : `tuple`
            The bounding points for the continuous domain. The bounds will
            be included as discrete points in the :py:class:`ContinuousSet`
            but will not be used to restrict points added to the
            :py:class:`ContinuousSet` through the 'initialize' argument,
            a data file, or the add() method

        Attributes
        ----------
        _changed : `boolean`
            This keeps track of whether or not the ContinuousSet was changed
            during discretization. If the user specifies all of the needed
            discretization points before the discretization then there is no
            need to go back through the model and reconstruct things indexed
            by the :py:class:`ContinuousSet`

        _fe : `list`
            This is a sorted list of the finite element points in the
            :py:class:`ContinuousSet`. i.e. this list contains all the
            discrete points in the :py:class:`ContinuousSet` that are not
            collocation points. Points that are both finite element points
            and collocation points will be included in this list.

        _discretization_info : `dict`
            This is a dictionary which contains information on the
            discretization transformation which has been applied to the
            :py:class:`ContinuousSet`.
    """

    def __init__(self, *args, **kwds):
        """ Constructor """
        if kwds.pop("filter", None) is not None:
            raise TypeError("'filter' is not a valid keyword argument for "
                            "ContinuousSet")
        # if kwds.pop("within", None) is not None:
        #    raise TypeError("'within' is not a valid keyword argument for "
        #  ContinuousSet")
        if kwds.pop("dimen", None) is not None:
            raise TypeError("'dimen' is not a valid keyword argument for "
                            "ContinuousSet")
        if kwds.pop("virtual", None) is not None:
            raise TypeError("'virtual' is not a valid keyword argument for "
                            "ContinuousSet")
        if kwds.pop("validate", None) is not None:
            raise TypeError("'validate' is not a valid keyword argument for "
                            "ContinuousSet")
        if len(args) != 0:
            raise TypeError("A ContinuousSet expects no arguments")

        kwds.setdefault('ctype', ContinuousSet)
        kwds.setdefault('ordered', Set.SortedOrder)
        self._type = ContinuousSet
        self._changed = False
        self.concrete = True
        self.virtual = False
        self._fe = []
        self._discretization_info = {}
        OrderedSimpleSet.__init__(self, **kwds)

    def get_finite_elements(self):
        """ Returns the finite element points

        If the :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>` has been
        discretizaed using a collocation scheme, this method will return a
        list of the finite element discretization points but not the
        collocation points within each finite element. If the
        :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>` has not been
        discretized or a finite difference discretization was used,
        this method returns a list of all the discretization points in the
        :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>`.

        Returns
        -------
        `list` of `floats`
        """
        return self._fe

    def get_discretization_info(self):
        """Returns a `dict` with information on the discretization scheme
        that has been applied to the :py:class:`ContinuousSet`.

        Returns
        -------
        `dict`
        """
        return self._discretization_info

    def get_changed(self):
        """ Returns flag indicating if the :py:class:`ContinuousSet` was
        changed during discretization

        Returns "True" if additional points were added to the
        :py:class:`ContinuousSet <pyomo.dae.ContinuousSet>` while applying a
        discretization scheme

        Returns
        -------
        `boolean`
        """
        return self._changed

    def set_changed(self, newvalue):
        """ Sets the ``_changed`` flag to 'newvalue'

        Parameters
        ----------
        newvalue : `boolean`

        """
        # TODO: Check this if-statement
        if newvalue is not True and newvalue is not False:
            raise ValueError("The _changed attribute on a ContinuousSet may "
                             "only be set to True or False")
        self._changed = newvalue

    def get_upper_element_boundary(self, point):
        """ Returns the first finite element point that is greater or equal
        to 'point'

        Parameters
        ----------
        point : `float`

        Returns
        -------
        float
        """
        if point in self._fe:
            return point
        elif point > max(self._fe):
            logger.warn("The point '%s' exceeds the upper bound "
                        "of the ContinuousSet '%s'. Returning the upper bound"
                        % (str(point), self.name))
            return max(self._fe)
        else:
            for i in self._fe:
                # This works because the list _fe is always sorted
                if i > point:
                    return i

    def get_lower_element_boundary(self, point):
        """ Returns the first finite element point that is less than or
        equal to 'point'

        Parameters
        ----------
        point : `float`

        Returns
        -------
        float
        """
        if point in self._fe:
            if 'scheme' in self._discretization_info:
                if self._discretization_info['scheme'] == 'LAGRANGE-RADAU':
                    # Because Radau Collocation has a collocation point on the
                    # upper finite element bound this if statement ensures that
                    # the desired finite element bound is returned
                    tmp = self._fe.index(point)
                    if tmp != 0:
                        return self._fe[tmp - 1]
            return point
        elif point < min(self._fe):
            logger.warn("The point '%s' is less than the lower bound "
                        "of the ContinuousSet '%s'. Returning the lower bound "
                        % (str(point), self.name))
            return min(self._fe)
        else:
            rev_fe = list(self._fe)
            rev_fe.reverse()
            for i in rev_fe:
                if i < point:
                    return i

    def construct(self, values=None):
        """ Constructs a :py:class:`ContinuousSet` component

        """
        timer = ConstructionTimer(self)
        OrderedSimpleSet.construct(self, values)

        for val in self.value:
            if type(val) is tuple:
                raise ValueError("ContinuousSet cannot contain tuples")
            if val.__class__ not in native_numeric_types:
                raise ValueError("ContinuousSet can only contain numeric "
                                 "values")

        if self._bounds is None:
            raise ValueError("ContinuousSet '%s' must have at least two values"
                             " indicating the range over which a differential "
                             "equation is to be discretized" % self.name)

        # If bounds were set using pyomo parameters, get their values
        lb = value(self._bounds[0])
        ub = value(self._bounds[1])
        self._bounds = (lb, ub)

        if self._bounds[0].__class__ not in native_numeric_types:
            raise ValueError("Bounds on ContinuousSet must be numeric values")
        if self._bounds[1].__class__ not in native_numeric_types:
            raise ValueError("Bounds on ContinuousSet must be numeric values")

        # TBD: If a user specifies bounds they will be added to the set
        # unless the user specified bounds have been overwritten during
        # OrderedSimpleSet construction. This can lead to some unintuitive
        # behavior when the ContinuousSet is both initialized with values and
        # bounds are specified. The current implementation is consistent
        # with how 'Set' treats this situation.
        if self._bounds[0] not in self.value:
            self.add(self._bounds[0])
            self._sort()
        if self._bounds[1] not in self.value:
            self.add(self._bounds[1])
            self._sort()

        if len(self) < 2:
            raise ValueError("ContinuousSet '%s' must have at least two values"
                             " indicating the range over which a differential "
                             "equation is to be discretized" % self.name)
        self._fe = sorted(self)
        timer.report()

