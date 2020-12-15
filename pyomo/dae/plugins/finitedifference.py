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

from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral

from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.diffvar import DAE_Error

from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In

logger = logging.getLogger('pyomo.dae')


def _central_transform(v, s):
    """
    Applies the Central Difference formula of order O(h^2) for first
    derivatives
    """
    def _ctr_fun(i):
        tmp = list(s)
        idx = s.ord(i)-1
        if idx == 0:  # Needed since '-1' is considered a valid index in Python
            raise IndexError("list index out of range")
        return 1 / (tmp[idx + 1] - tmp[idx - 1]) * \
               (v(tmp[idx + 1]) - v(tmp[idx - 1]))
    return _ctr_fun


def _central_transform_order2(v, s):
    """
    Applies the Central Difference formula of order O(h^2) for second
    derivatives
    """
    def _ctr_fun2(i):
        tmp = list(s)
        idx = s.ord(i)-1
        if idx == 0:  # Needed since '-1' is considered a valid index in Python
            raise IndexError("list index out of range")
        return 1 / ((tmp[idx + 1] - tmp[idx]) * (tmp[idx] - tmp[idx - 1])) * \
               (v(tmp[idx + 1]) - 2 * v(tmp[idx]) + v(tmp[idx - 1]))
    return _ctr_fun2


def _forward_transform(v, s):
    """
    Applies the Forward Difference formula of order O(h) for first derivatives
    """
    def _fwd_fun(i):
        tmp = list(s)
        idx = s.ord(i)-1
        return 1 / (tmp[idx + 1] - tmp[idx]) * (v(tmp[idx + 1]) - v(tmp[idx]))
    return _fwd_fun


def _forward_transform_order2(v, s):
    """
    Applies the Forward Difference formula of order O(h) for second derivatives
    """
    def _fwd_fun(i):
        tmp = list(s)
        idx = s.ord(i)-1
        return 1 / ((tmp[idx + 2] - tmp[idx + 1]) *
                    (tmp[idx + 1] - tmp[idx])) *\
               (v(tmp[idx + 2]) - 2 * v(tmp[idx + 1]) + v(tmp[idx]))
    return _fwd_fun


def _backward_transform(v, s):
    """
    Applies the Backward Difference formula of order O(h) for first derivatives
    """
    def _bwd_fun(i):
        tmp = list(s)
        idx = s.ord(i)-1
        if idx == 0:  # Needed since '-1' is considered a valid index in Python
            raise IndexError("list index out of range")
        return 1 / (tmp[idx] - tmp[idx - 1]) * (v(tmp[idx]) - v(tmp[idx - 1]))
    return _bwd_fun


def _backward_transform_order2(v, s):
    """
    Applies the Backward Difference formula of order O(h) for second
    derivatives
    """
    def _bwd_fun(i):
        tmp = list(s)
        idx = s.ord(i)-1

        # This check is needed since '-1' is considered a valid index in Python
        if idx == 0 or idx == 1:
            raise IndexError("list index out of range")
        return 1 / ((tmp[idx - 1] - tmp[idx - 2]) *
                    (tmp[idx] - tmp[idx - 1])) * \
               (v(tmp[idx]) - 2 * v(tmp[idx - 1]) + v(tmp[idx - 2]))
    return _bwd_fun


@TransformationFactory.register('dae.finite_difference', doc="Discretizes a DAE model using "
          "a finite difference method transforming the model into an NLP.")
class Finite_Difference_Transformation(Transformation):
    """
    Transformation that applies finite difference methods to
    DAE, ODE, or PDE models.
    """

    CONFIG = ConfigBlock("dae.finite_difference")
    CONFIG.declare('nfe', ConfigValue(
        default=10,
        domain=PositiveInt,
        description="The desired number of finite element points to be "
                    "included in the discretization"
    ))
    CONFIG.declare('wrt', ConfigValue(
        default=None,
        description="The ContinuousSet to be discretized",
        doc="Indicates which ContinuousSet the transformation should be "
            "applied to. If this keyword argument is not specified then the "
            "same scheme will be applied to all ContinuousSets."
    ))
    CONFIG.declare('scheme', ConfigValue(
        default='BACKWARD',
        domain=In(['BACKWARD', 'CENTRAL', 'FORWARD']),
        description="Indicates which finite difference scheme to apply",
        doc="Options are BACKWARD, CENTRAL, or FORWARD. The default scheme is "
            "the backward difference method"
    ))

    def __init__(self):
        super(Finite_Difference_Transformation, self).__init__()
        self._nfe = {}
        self.all_schemes = {
            'BACKWARD': (_backward_transform, _backward_transform_order2),
            'CENTRAL': (_central_transform, _central_transform_order2),
            'FORWARD': (_forward_transform, _forward_transform_order2)}

    def _apply_to(self, instance, **kwds):
        """
        Applies the transformation to a modeling instance

        Keyword Arguments:
        nfe           The desired number of finite element points to be
                      included in the discretization.
        wrt           Indicates which ContinuousSet the transformation
                      should be applied to. If this keyword argument is not
                      specified then the same scheme will be applied to all
                      ContinuousSets.
        scheme        Indicates which finite difference method to apply.
                      Options are BACKWARD, CENTRAL, or FORWARD. The default
                      scheme is the backward difference method
        """

        config = self.CONFIG(kwds)

        tmpnfe = config.nfe
        tmpds = config.wrt

        if tmpds is not None:
            if tmpds.ctype is not ContinuousSet:
                raise TypeError("The component specified using the 'wrt' "
                                "keyword must be a continuous set")
            elif 'scheme' in tmpds.get_discretization_info():
                raise ValueError("The discretization scheme '%s' has already "
                                 "been applied to the ContinuousSet '%s'" %
                                 (tmpds.get_discretization_info()['scheme'],
                                  tmpds.name))

        if None in self._nfe:
            raise ValueError(
                "A general discretization scheme has already been applied to "
                "to every continuous set in the model. If you would like to "
                "apply a different discretization scheme to each continuous "
                "set, you must declare a new transformation object")

        if len(self._nfe) == 0 and tmpds is None:
            # Same discretization on all ContinuousSets
            self._nfe[None] = tmpnfe
            currentds = None
        else:
            self._nfe[tmpds.name] = tmpnfe
            currentds = tmpds.name

        self._scheme_name = config.scheme
        self._scheme = self.all_schemes.get(self._scheme_name, None)

        self._transformBlock(instance, currentds)

        return instance

    def _transformBlock(self, block, currentds):

        self._fe = {}
        for ds in block.component_objects(ContinuousSet):
            if currentds is None or currentds == ds.name or currentds is ds:
                if 'scheme' in ds.get_discretization_info():
                    raise DAE_Error("Attempting to discretize ContinuousSet "
                                    "'%s' after it has already been discretized. "
                                    % ds.name)
                generate_finite_elements(ds, self._nfe[currentds])
                if not ds.get_changed():
                    if len(ds) - 1 > self._nfe[currentds]:
                        logger.warn("More finite elements were found in "
                                    "ContinuousSet '%s' than the number of "
                                    "finite elements specified in apply. The "
                                    "larger number of finite elements will be "
                                    "used." % ds.name)

                self._nfe[ds.name] = len(ds) - 1
                self._fe[ds.name] = list(ds)
                # Adding discretization information to the ContinuousSet
                # object itself so that it can be accessed outside of the
                # discretization object
                disc_info = ds.get_discretization_info()
                disc_info['nfe'] = self._nfe[ds.name]
                disc_info['scheme'] = self._scheme_name + ' Difference'

        # Maybe check to see if any of the ContinuousSets have been changed,
        # if they haven't then the model components need not be updated
        # or even iterated through
        expand_components(block)

        for d in block.component_objects(DerivativeVar, descend_into=True):
            dsets = d.get_continuousset_list()
            for i in ComponentSet(dsets):
                if currentds is None or i.name == currentds:
                    oldexpr = d.get_derivative_expression()
                    loc = d.get_state_var()._contset[i]
                    count = dsets.count(i)
                    if count >= 3:
                        raise DAE_Error(
                            "Error discretizing '%s' with respect to '%s'. "
                            "Current implementation only allows for taking the"
                            " first or second derivative with respect to "
                            "a particular ContinuousSet" % (d.name, i.name))
                    scheme = self._scheme[count - 1]
                    newexpr = create_partial_expression(scheme, oldexpr, i,
                                                        loc)
                    d.set_derivative_expression(newexpr)

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
                # If a model contains integrals they are most likely to
                # appear in the objective function which will need to be
                # reconstructed after the model is discretized.
                for k in block.component_objects(Objective, descend_into=True):
                    # TODO: check this, reconstruct might not work
                    k.reconstruct()
