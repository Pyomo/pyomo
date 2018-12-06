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
from six.moves import xrange
from six import next

from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral

from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error

# If the user has numpy then the collocation points and the a matrix for
# the Runge-Kutta basis formulation will be calculated as needed.
# If the user does not have numpy then these values will be read from a
# stored dictionary for up to 10 collocation points.
try:
    import numpy
    numpy_available = True
except ImportError:  # pragma:nocover
    numpy_available = False

logger = logging.getLogger('pyomo.dae')


def _lagrange_radau_transform(v, s):
    ncp = s.get_discretization_info()['ncp']
    adot = s.get_discretization_info()['adot']

    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0:  # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx + j]) * adot[j][idx - lowidx] *
                   (1.0 / (tmp[lowidx + ncp] - tmp[lowidx]))
                   for j in range(ncp + 1))
    return _fun


def _lagrange_radau_transform_order2(v, s):
    ncp = s.get_discretization_info()['ncp']
    adotdot = s.get_discretization_info()['adotdot']

    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0:  # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx + j]) * adotdot[j][idx - lowidx] *
                   (1.0 / (tmp[lowidx + ncp] - tmp[lowidx]) ** 2)
                   for j in range(ncp + 1))
    return _fun


def _lagrange_legendre_transform(v, s):
    ncp = s.get_discretization_info()['ncp']
    adot = s.get_discretization_info()['adot']

    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0:  # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        elif i in s.get_finite_elements():  # Don't apply at finite element
                                            # points continuity equations
                                            # added later
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx + j]) * adot[j][idx - lowidx] *
                   (1.0 / (tmp[lowidx + ncp + 1] - tmp[lowidx]))
                   for j in range(ncp + 1))
    return _fun


def _lagrange_legendre_transform_order2(v, s):
    ncp = s.get_discretization_info()['ncp']
    adotdot = s.get_discretization_info()['adotdot']

    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0:  # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        elif i in s.get_finite_elements():  # Don't apply at finite element
                                            # points continuity equations
                                            # added later
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx + j]) * adotdot[j][idx - lowidx] *
                   (1.0 / (tmp[lowidx + ncp + 1] - tmp[lowidx]) ** 2) \
                   for j in range(ncp + 1))
    return _fun


def conv(a, b):
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Cannot convolve an empty list")

    ans = []
    m = len(a)
    n = len(b)

    for k in range(m + n - 1):
        val = 0
        j = max(0, k - n)
        stop = min(k, m)
        while j <= stop:
            if j < m and (k - j) < n:
                val += a[j] * b[k - j]
            j += 1
        ans.insert(k, val)

    return ans


def calc_cp(alpha, beta, k):
    gamma = []
    factorial = numpy.math.factorial
    
    for i in range(k + 1):
        num = factorial(alpha + k) * factorial(alpha + beta + k + i)
        denom = factorial(alpha + i) * factorial(k - i) * factorial(i)
        gamma.insert(i, num / denom)

    poly = []
    for i in range(k + 1):
        if i == 0:
            poly.insert(i, gamma[i])
        else:
            prod = [1]
            j = 1
            while j <= i:
                prod = conv(prod, [1, -1])
                j += 1
            while len(poly) < len(prod):
                poly.insert(0, 0)
            prod = [gamma[i] * t for t in prod]
            poly = [sum(pair) for pair in zip(poly, prod)]

    cp = numpy.roots(poly)
    return cp

# BLN: This is a legacy function that was used to calculate the collocation
# constants for an alternative form of the collocation equations described
# in Biegler's nonlinear programming book. The difference being whether the 
# state or the derivative is approximated using lagrange polynomials. With 
# the addition of PDE support and chained discretizations in Pyomo.DAE 2.0
# this function is no longer used but kept here for future reference.
#
# def calc_omega(cp):
#     a = []
#     for i in range(len(cp)):
#         ptmp = []
#         tmp = 0
#         for j in range(len(cp)):
#             if j != i:
#                 row = []
#                 row.insert(0, 1 / (cp[i] - cp[j]))
#                 row.insert(1, -cp[j] / (cp[i] - cp[j]))
#                 ptmp.insert(tmp, row)
#                 tmp += 1
#         p = [1]
#         for j in range(len(cp) - 1):
#             p = conv(p, ptmp[j])
#         pint = numpy.polyint(p)
#         arow = []
#         for j in range(len(cp)):
#             arow.append(numpy.polyval(pint, cp[j]))
#         a.append(arow)
#     return a


def calc_adot(cp, order=1):
    a = []
    for i in range(len(cp)):
        ptmp = []
        tmp = 0
        for j in range(len(cp)):
            if j != i:
                row = []
                row.insert(0, 1 / (cp[i] - cp[j]))
                row.insert(1, -cp[j] / (cp[i] - cp[j]))
                ptmp.insert(tmp, row)
                tmp += 1
        p = [1]
        for j in range(len(cp) - 1):
            p = conv(p, ptmp[j])
        pder = numpy.polyder(p, order)
        arow = []
        for j in range(len(cp)):
            arow.append(numpy.polyval(pder, cp[j]))
        a.append(arow)
    return a


def calc_afinal(cp):
    afinal = []
    for i in range(len(cp)):
        ptmp = []
        tmp = 0
        for j in range(len(cp)):
            if j != i:
                row = []
                row.insert(0, 1 / (cp[i] - cp[j]))
                row.insert(1, -cp[j] / (cp[i] - cp[j]))
                ptmp.insert(tmp, row)
                tmp += 1
        p = [1]
        for j in range(len(cp) - 1):
            p = conv(p, ptmp[j])
        afinal.append(numpy.polyval(p, 1.0))
    return afinal


@TransformationFactory.register('dae.collocation', 
            doc="Discretizes a DAE model using "
            "orthogonal collocation over finite elements transforming "
            "the model into an NLP.")
class Collocation_Discretization_Transformation(Transformation):

    def __init__(self):
        super(Collocation_Discretization_Transformation, self).__init__()
        self._ncp = {}
        self._nfe = {}
        self._adot = {}
        self._adotdot = {}
        self._afinal = {}
        self._tau = {}
        self._reduced_cp = {}
        self.all_schemes = {
            'LAGRANGE-RADAU': (_lagrange_radau_transform,
                               _lagrange_radau_transform_order2),
            'LAGRANGE-LEGENDRE': (_lagrange_legendre_transform,
                                  _lagrange_legendre_transform_order2)}

    def _get_radau_constants(self, currentds):
        """
        This function sets the radau collocation points and a values depending
        on how many collocation points have been specified and whether or not
        the user has numpy
        """
        if not numpy_available:
            if self._ncp[currentds] > 10:
                raise ValueError("Numpy was not found so the maximum number "
                                 "of collocation points is 10")
            from pyomo.dae.utilities import (radau_tau_dict, radau_adot_dict,
                                             radau_adotdot_dict)
            self._tau[currentds] = radau_tau_dict[self._ncp[currentds]]
            self._adot[currentds] = radau_adot_dict[self._ncp[currentds]]
            self._adotdot[currentds] = radau_adotdot_dict[self._ncp[currentds]]
            self._afinal[currentds] = None
        else:
            alpha = 1
            beta = 0
            k = self._ncp[currentds] - 1
            cp = sorted(list(calc_cp(alpha, beta, k)))
            cp.insert(0, 0.0)
            cp.append(1.0)
            adot = calc_adot(cp, 1)
            adotdot = calc_adot(cp, 2)

            self._tau[currentds] = cp
            self._adot[currentds] = adot
            self._adotdot[currentds] = adotdot
            self._afinal[currentds] = None

    def _get_legendre_constants(self, currentds):
        """
        This function sets the legendre collocation points and a values
        depending on how many collocation points have been specified and
        whether or not the user has numpy
        """
        if not numpy_available:
            if self._ncp[currentds] > 10:
                raise ValueError("Numpy was not found so the maximum number "
                                 "of collocation points is 10")
            from pyomo.dae.utilities import (legendre_tau_dict,
                                             legendre_adot_dict,
                                             legendre_adotdot_dict,
                                             legendre_afinal_dict)
            self._tau[currentds] = legendre_tau_dict[self._ncp[currentds]]
            self._adot[currentds] = legendre_adot_dict[self._ncp[currentds]]
            self._adotdot[currentds] = \
                legendre_adotdot_dict[self._ncp[currentds]]
            self._afinal[currentds] = \
                legendre_afinal_dict[self._ncp[currentds]]
        else:
            alpha = 0
            beta = 0
            k = self._ncp[currentds]
            cp = sorted(list(calc_cp(alpha, beta, k)))
            cp.insert(0, 0.0)
            adot = calc_adot(cp, 1)
            adotdot = calc_adot(cp, 2)
            afinal = calc_afinal(cp)

            self._tau[currentds] = cp
            self._adot[currentds] = adot
            self._adotdot[currentds] = adotdot
            self._afinal[currentds] = afinal

    def _apply_to(self, instance, **kwds):
        """
        Applies specified collocation transformation to a modeling instance

        Keyword Arguments:
        nfe           The desired number of finite element points to be
                      included in the discretization.
        ncp           The desired number of collocation points over each
                      finite element.
        wrt           Indicates which ContinuousSet the transformation
                      should be applied to. If this keyword argument is not
                      specified then the same scheme will be applied to all
                      ContinuousSets.
        scheme        Indicates which finite difference method to apply.
                      Options are 'LAGRANGE-RADAU' and 'LAGRANGE-LEGENDRE'. 
                      The default scheme is Lagrange polynomials with Radau
                      roots.
        """

        tmpnfe = kwds.pop('nfe', 10)
        tmpncp = kwds.pop('ncp', 3)
        tmpds = kwds.pop('wrt', None)
        tmpscheme = kwds.pop('scheme', 'LAGRANGE-RADAU')
        self._scheme_name = tmpscheme.upper()

        if tmpds is not None:
            if tmpds.type() is not ContinuousSet:
                raise TypeError("The component specified using the 'wrt' "
                                "keyword must be a continuous set")
            elif 'scheme' in tmpds.get_discretization_info():
                raise ValueError("The discretization scheme '%s' has already "
                                 "been applied to the ContinuousSet '%s'"
                                 % (tmpds.get_discretization_info()['scheme'],
                                    tmpds.name))

        if tmpnfe <= 0:
            raise ValueError(
                "The number of finite elements must be at least 1")
        if tmpncp <= 0:
            raise ValueError(
                "The number of collocation points must be at least 1")

        if None in self._nfe:
            raise ValueError(
                "A general discretization scheme has already been applied to "
                "to every ContinuousSet in the model. If you would like to "
                "specify a specific discretization scheme for one of the "
                "ContinuousSets you must discretize each ContinuousSet "
                "separately.")

        if len(self._nfe) == 0 and tmpds is None:
            # Same discretization on all ContinuousSets
            self._nfe[None] = tmpnfe
            self._ncp[None] = tmpncp
            currentds = None
        else:
            self._nfe[tmpds.name] = tmpnfe
            self._ncp[tmpds.name] = tmpncp
            currentds = tmpds.name

        self._scheme = self.all_schemes.get(self._scheme_name, None)
        if self._scheme is None:
            raise ValueError("Unknown collocation scheme '%s' specified using "
                             "the 'scheme' keyword. Valid schemes are "
                             "'LAGRANGE-RADAU' and 'LAGRANGE-LEGENDRE'"
                              % tmpscheme)

        if self._scheme_name == 'LAGRANGE-RADAU':
            self._get_radau_constants(currentds)
        elif self._scheme_name == 'LAGRANGE-LEGENDRE':
            self._get_legendre_constants(currentds)

        self._transformBlock(instance, currentds)

        return instance

    def _transformBlock(self, block, currentds):

        self._fe = {}
        for ds in block.component_objects(ContinuousSet, descend_into=True):
            if currentds is None or currentds == ds.name:
                generate_finite_elements(ds, self._nfe[currentds])
                if not ds.get_changed():
                    if len(ds) - 1 > self._nfe[currentds]:
                        logger.warn("More finite elements were found in "
                                    "ContinuousSet '%s' than the number of "
                                    "finite elements specified in apply. The "
                                    "larger number of finite elements will be "
                                    "used." % ds.name)

                self._nfe[ds.name] = len(ds) - 1
                self._fe[ds.name] = sorted(ds)
                generate_colloc_points(ds, self._tau[currentds])
                # Adding discretization information to the continuousset
                # object itself so that it can be accessed outside of the
                # discretization object
                disc_info = ds.get_discretization_info()
                disc_info['nfe'] = self._nfe[ds.name]
                disc_info['ncp'] = self._ncp[currentds]
                disc_info['tau_points'] = self._tau[currentds]
                disc_info['adot'] = self._adot[currentds]
                disc_info['adotdot'] = self._adotdot[currentds]
                disc_info['afinal'] = self._afinal[currentds]
                disc_info['scheme'] = self._scheme_name

        expand_components(block)

        for d in block.component_objects(DerivativeVar, descend_into=True):
            dsets = d.get_continuousset_list()
            for i in set(dsets):
                if currentds is None or i.name == currentds:
                    oldexpr = d.get_derivative_expression()
                    loc = d.get_state_var()._contset[i]
                    count = dsets.count(i)
                    if count >= 3:
                        raise DAE_Error(
                            "Error discretizing '%s' with respect to '%s'. "
                            "Current implementation only allows for taking the"
                            " first or second derivative with respect to a "
                            "particular ContinuousSet" % (d.name, i.name))
                    scheme = self._scheme[count - 1]

                    newexpr = create_partial_expression(scheme, oldexpr, i,
                                                        loc)
                    d.set_derivative_expression(newexpr)
                    if self._scheme_name == 'LAGRANGE-LEGENDRE':
                        # Add continuity equations to DerivativeVar's parent
                        #  block
                        add_continuity_equations(d.parent_block(), d, i, loc)

            # Reclassify DerivativeVar if all indexing ContinuousSets have
            # been discretized. Add discretization equations to the
            # DerivativeVar's parent block.
            if d.is_fully_discretized():
                add_discretization_equations(d.parent_block(), d)
                d.parent_block().reclassify_component_type(d, Var)

                # Keep track of any reclassified DerivativeVar components so
                # that the Simulator can easily identify them if the model
                # is simulated after discretization
                # TODO: Update the discretization transformations to use
                # a Block to add things to the model and store discretization
                # information. Using a list for now because the simulator
                # does not yet support models containing active Blocks
                reclassified_list = getattr(block,
                                            '_pyomo_dae_reclassified_derivativevars',
                                            None)
                if reclassified_list is None:
                    block._pyomo_dae_reclassified_derivativevars = list()
                    reclassified_list = \
                        block._pyomo_dae_reclassified_derivativevars

                reclassified_list.append(d)

        # Reclassify Integrals if all ContinuousSets have been discretized
        if block_fully_discretized(block):

            if block.contains_component(Integral):
                for i in block.component_objects(Integral, descend_into=True):
                    i.reconstruct()
                    i.parent_block().reclassify_component_type(i, Expression)
                # If a model contains integrals they are most likely to appear
                # in the objective function which will need to be reconstructed
                # after the model is discretized.
                for k in block.component_objects(Objective, descend_into=True):
                    # TODO: check this, reconstruct might not work
                    k.reconstruct()

    def _get_idx(self, l, t, n, i, k):
        """
        This function returns the appropriate index for the ContinuousSet
        and the derivative variables. It's needed because the collocation
        constraints are indexed by finite element and collocation point
        however a ContinuousSet contains a list of all the discretization
        points and is not separated into finite elements and collocation
        points.
        """

        tmp = t.index(t._fe[i])
        tik = t[tmp + k]
        if n is None:
            return tik
        else:
            tmpn = n
            if not isinstance(n, tuple):
                tmpn = (n,)
            return tmpn[0:l] + (tik,) + tmpn[l:]

    def reduce_collocation_points(self, instance, var=None, ncp=None,
                                  contset=None):
        """
        This method will add additional constraints to a model to reduce the
        number of free collocation points (degrees of freedom) for a particular
        variable.

        Parameters
        ----------
        instance : Pyomo model
            The discretized Pyomo model to add constraints to

        var : ``pyomo.environ.Var``
            The Pyomo variable for which the degrees of freedom will be reduced

        ncp : int
            The new number of free collocation points for `var`. Must be
            less that the number of collocation points used in discretizing
            the model.

        contset : ``pyomo.dae.ContinuousSet``
            The :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` that was
            discretized and for which the `var` will have a reduced number
            of degrees of freedom

        """
        if contset is None:
            raise TypeError("A continuous set must be specified using the "
                            "keyword 'contset'")
        if contset.type() is not ContinuousSet:
            raise TypeError("The component specified using the 'contset' "
                            "keyword must be a ContinuousSet")
        ds = contset

        if len(self._ncp) == 0:
            raise RuntimeError("This method should only be called after using "
                               "the apply() method to discretize the model")
        elif None in self._ncp:
            tot_ncp = self._ncp[None]
        elif ds.name in self._ncp:
            tot_ncp = self._ncp[ds.name]
        else:
            raise ValueError("ContinuousSet '%s' has not been discretized, "
                             "please call the apply_to() method with this "
                             "ContinuousSet to discretize it before calling "
                             "this method" % ds.name)

        if var is None:
            raise TypeError("A variable must be specified")
        if var.type() is not Var:
            raise TypeError("The component specified using the 'var' keyword "
                            "must be a variable")

        if ncp is None:
            raise TypeError(
                "The number of collocation points must be specified")
        if ncp <= 0:
            raise ValueError(
                "The number of collocation points must be at least 1")
        if ncp > tot_ncp:
            raise ValueError("The number of collocation points used to "
                             "interpolate an individual variable must be less "
                             "than the number used to discretize the original "
                             "model")
        if ncp == tot_ncp:
            # Nothing to be done
            return instance

        # Check to see if the continuousset is an indexing set of the variable
        if var.dim() == 0:
            raise IndexError("ContinuousSet '%s' is not an indexing set of"
                             " the variable '%s'" % (ds.name, var.name))
        elif var.dim() == 1:
            if ds not in var._index:
                raise IndexError("ContinuousSet '%s' is not an indexing set of"
                                 " the variable '%s'" % (ds.name, var.name))
        elif ds not in var._implicit_subsets:
            raise IndexError("ContinuousSet '%s' is not an indexing set of the"
                             " variable '%s'" % (ds.name, var.name))

        if var.name in self._reduced_cp:
            temp = self._reduced_cp[var.name]
            if ds.name in temp:
                raise RuntimeError("Variable '%s' has already been constrained"
                                   " to a reduced number of collocation points"
                                   " over ContinuousSet '%s'.")
            else:
                temp[ds.name] = ncp
        else:
            self._reduced_cp[var.name] = {ds.name: ncp}

        # TODO: Use unique_component_name for this
        list_name = var.local_name + "_interpolation_constraints"

        instance.add_component(list_name, ConstraintList())
        conlist = instance.find_component(list_name)

        t = sorted(ds)
        fe = ds._fe
        info = get_index_information(var, ds)
        tmpidx = info['non_ds']
        idx = info['index function']

        # Iterate over non_ds indices
        for n in tmpidx:
            # Iterate over finite elements
            for i in xrange(0, len(fe) - 1):
                # Iterate over collocation points
                for k in xrange(1, tot_ncp - ncp + 1):
                    if ncp == 1:
                        # Constant over each finite element
                        conlist.add(var[idx(n, i, k)] ==
                                    var[idx(n, i, tot_ncp)])
                    else:
                        tmp = t.index(fe[i])
                        tmp2 = t.index(fe[i + 1])
                        ti = t[tmp + k]
                        tfit = t[tmp2 - ncp + 1:tmp2 + 1]
                        coeff = self._interpolation_coeffs(ti, tfit)
                        conlist.add(var[idx(n, i, k)] ==
                                    sum(var[idx(n, i, j)] * next(coeff)
                                        for j in xrange(tot_ncp - ncp + 1,
                                                        tot_ncp + 1)))

        return instance

    def _interpolation_coeffs(self, ti, tfit):

        for i in tfit:
            l = 1
            for j in tfit:
                if i != j:
                    l = l * (ti - j) / (i - j)
            yield l
