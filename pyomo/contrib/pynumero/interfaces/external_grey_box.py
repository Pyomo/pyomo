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

import abc
import logging
from scipy.sparse import coo_matrix
from pyomo.common.dependencies import numpy as np

from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base import Var, Set, Constraint, value
from pyomo.core.base.block import _BlockData, Block, declare_custom_block
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.set import UnindexedComponent_set
from pyomo.core.base.reference import Reference

from ..sparse.block_matrix import BlockMatrix


logger = logging.getLogger('pyomo.contrib.pynumero')

"""
This module is used for interfacing an external model as
a block in a Pyomo model.

An ExternalGreyBoxModel is a model that does not
provide constraints explicitly as algebraic expressions, but
instead provides a set of methods that can compute the residuals
of the constraints (or outputs) and their derivatives.

This allows one to interface external codes (e.g., compiled
external models) with a Pyomo model.

Note: To solve a Pyomo model that contains these external models
      we have a specialized interface built on PyNumero that provides
      an interface to the CyIpopt solver.

To use this interface:
   * Create a class that is derived from ExternalGreyBoxModel and
     implement the necessary methods. This derived class must provide
     a list of names for: the inputs to your model, the equality constraints
     (or residuals) that need to be converged, and any outputs that
     are computed from your model. It will also need to provide methods to
     compute the residuals, outputs, and the jacobian of these with respect to
     the inputs. Implement the methods to evaluate hessians if applicable.
     See the documentation on ExternalGreyBoxModel for more details.

   * Create a Pyomo model and make use of the ExternalGreyBoxBlock
     to produce a Pyomo modeling component that represents your
     external model. This block is a Pyomo component, and when you
     call set_external_model() and provide an instance of your derived
     ExternalGreyBoxModel, it will automatically create pyomo variables to
     represent the inputs and the outputs from the external model. You
     can implement a callback to modify the Pyomo block after it is
     constructed. This also provides a mechanism to initialize variables,
     etc.

   * Create a PyomoGreyBoxNLP and provide it with the Pyomo model
     that contains the ExternalGreyBoxBlocks. This class presents
     an NLP interface (i.e., the PyNumero NLP abstract class), and
     can be used with any solver that makes use of this interface
     (e.g., the CyIpopt solver interface provided in PyNumero)

See pyomo/contrib/pynumero/examples/external_grey_box for examples
of the use of this interface.

Note:

   * Currently, you cannot "fix" a pyomo variable that corresponds to an
     input or output and you must use a constraint instead (this is
     because Pyomo removes fixed variables before sending them to the
     solver)

"""


class ExternalGreyBoxModel(object):
    """
    This is the base class for building external input output models
    for use with Pyomo and CyIpopt. See the module documentation above,
    and documentation of individual methods.

    There are examples in:
    pyomo/contrib/pynumero/examples/external_grey_box/react-example/

    Most methods are documented in the class itself. However, there are
    methods that are not implemented in the base class that may need
    to be implemented to provide support for certain features.

    Hessian support:

    If you would like to support Hessian computations for your
    external model, you will need to implement the following methods to
    support setting the multipliers that are used when computing the
    Hessian of the Lagrangian.
    - set_equality_constraint_multipliers: see documentation in method
    - set_output_constraint_multipliers: see documentation in method
    You will also need to implement the following methods to evaluate
    the required Hessian information:

    def evaluate_hessian_equality_constraints(self):
        Compute the product of the equality constraint multipliers
        with the hessian of the equality constraints.
        E.g., y_eq^k is the vector of equality constraint multipliers
        from set_equality_constraint_multipliers, w_eq(u)=0 are the
        equality constraints, and u^k are the vector of inputs from
        set_inputs. This method must return
        H_eq^k = sum_i (y_eq^k)_i * grad^2_{uu} w_eq(u^k)

    def evaluate_hessian_outputs(self):
        Compute the product of the output constraint multipliers with the
        hessian of the outputs. E.g., y_o^k is the vector of output
        constraint multipliers from set_output_constraint_multipliers,
        u^k are the vector of inputs from set_inputs, and w_o(u) is the
        function that computes the vector of outputs at the values for
        the input variables. This method must return
        H_o^k = sum_i (y_o^k)_i * grad^2_{uu} w_o(u^k)

    Examples that show Hessian support are also found in:
    pyomo/contrib/pynumero/examples/external_grey_box/react-example/

    """

    def n_inputs(self):
        """This method returns the number of inputs. You do not
        need to overload this method in derived classes.
        """
        return len(self.input_names())

    def n_equality_constraints(self):
        """This method returns the number of equality constraints.
        You do not need to overload this method in derived classes.
        """
        return len(self.equality_constraint_names())

    def n_outputs(self):
        """This method returns the number of outputs. You do not
        need to overload this method in derived classes.
        """
        return len(self.output_names())

    def input_names(self):
        """
        Provide the list of string names to corresponding to the inputs
        of this external model. These should be returned in the same order
        that they are to be used in set_input_values.
        """
        raise NotImplementedError(
            'Derived ExternalGreyBoxModel classes need to implement the method: input_names'
        )

    def equality_constraint_names(self):
        """
        Provide the list of string names corresponding to any residuals
        for this external model. These should be in the order corresponding
        to values returned from evaluate_residuals. Return an empty list
        if there are no equality constraints.
        """
        return []

    def output_names(self):
        """
        Provide the list of string names corresponding to the outputs
        of this external model. These should be in the order corresponding
        to values returned from evaluate_outputs. Return an empty list if there
        are no computed outputs.
        """
        return []

    def finalize_block_construction(self, pyomo_block):
        """
        Implement this callback to provide any additional
        specifications to the Pyomo block that is created
        to represent this external grey box model.

        Note that pyomo_block.inputs and pyomo_block.outputs
        have been created, and this callback provides an
        opportunity to set initial values, bounds, etc.
        """
        pass

    def set_input_values(self, input_values):
        """
        This method is called by the solver to set the current values
        for the input variables. The derived class must cache these if
        necessary for any subsequent calls to evaluate_outputs or
        evaluate_derivatives.
        """
        raise NotImplementedError(
            'Derived ExternalGreyBoxModel classes need'
            ' to implement the method: set_input_values'
        )

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        """
        This method is called by the solver to set the current values
        for the multipliers of the equality constraints. The derived
        class must cache these if necessary for any subsequent calls
        to evaluate_hessian_equality_constraints
        """
        # we should check these for efficiency
        assert self.n_equality_constraints() == len(eq_con_multiplier_values)
        if (
            not hasattr(self, 'evaluate_hessian_equality_constraints')
            or self.n_equality_constraints() == 0
        ):
            return

        raise NotImplementedError(
            'Derived ExternalGreyBoxModel classes need to implement'
            ' set_equality_constraint_multipliers when they'
            ' support Hessian computations.'
        )

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        """
        This method is called by the solver to set the current values
        for the multipliers of the output constraints. The derived
        class must cache these if necessary for any subsequent calls
        to evaluate_hessian_outputs
        """
        # we should check these for efficiency
        assert self.n_outputs() == len(output_con_multiplier_values)
        if (
            not hasattr(self, 'evaluate_hessian_output_constraints')
            or self.n_outputs() == 0
        ):
            return

        raise NotImplementedError(
            'Derived ExternalGreyBoxModel classes need to implement'
            ' set_output_constraint_multipliers when they'
            ' support Hessian computations.'
        )

    def get_equality_constraint_scaling_factors(self):
        """
        This method is called by the solver interface to get desired
        values for scaling the equality constraints. None means no
        scaling is desired. Note that, depending on the solver,
        one may need to set solver options so these factors are used
        """
        return None

    def get_output_constraint_scaling_factors(self):
        """
        This method is called by the solver interface to get desired
        values for scaling the constraints with output variables. Returning
        None means that no scaling of the output constraints is desired.
        Note that, depending on the solver, one may need to set solver options
        so these factors are used
        """
        return None

    def evaluate_equality_constraints(self):
        """
        Compute the residuals from the model (using the values
        set in input_values) and return as a numpy array
        """
        raise NotImplementedError(
            'evaluate_equality_constraints called '
            'but not implemented in the derived class.'
        )

    def evaluate_outputs(self):
        """
        Compute the outputs from the model (using the values
        set in input_values) and return as a numpy array
        """
        raise NotImplementedError(
            'evaluate_outputs called but not implemented in the derived class.'
        )

    def evaluate_jacobian_equality_constraints(self):
        """
        Compute the derivatives of the residuals with respect
        to the inputs (using the values set in input_values).
        This should be a scipy matrix with the rows in
        the order of the residual names and the cols in
        the order of the input variables.
        """
        raise NotImplementedError(
            'evaluate_jacobian_equality_constraints called '
            'but not implemented in the derived class.'
        )

    def evaluate_jacobian_outputs(self):
        """
        Compute the derivatives of the outputs with respect
        to the inputs (using the values set in input_values).
        This should be a scipy matrix with the rows in
        the order of the output variables and the cols in
        the order of the input variables.
        """
        raise NotImplementedError(
            'evaluate_equality_outputs called '
            'but not implemented in the derived class.'
        )

    #
    # Implement the following methods to provide support for
    # Hessian computations: see documentation in class docstring
    #
    # def evaluate_hessian_equality_constraints(self):
    # def evaluate_hessian_outputs(self):
    #


class ExternalGreyBoxBlockData(_BlockData):
    def set_external_model(self, external_grey_box_model, inputs=None, outputs=None):
        """
        Parameters
        ----------
        external_grey_box_model: ExternalGreyBoxModel
            The external model that will be interfaced to in this block
        inputs: List of VarData objects
            If provided, these VarData will be used as inputs into the
            external model.
        outputs: List of VarData objects
            If provided, these VarData will be used as outputs from the
            external model.

        """
        self._ex_model = ex_model = external_grey_box_model
        if ex_model is None:
            self._input_names = self._output_names = None
            self.inputs = self.outputs = None
            return

        # Shouldn't need input names if we provide input vars, but
        # no reason to remove this.
        # _could_ make inputs a reference-to-mapping
        self._input_names = ex_model.input_names()
        if self._input_names is None or len(self._input_names) == 0:
            raise ValueError(
                'No input_names specified for external_grey_box_model.'
                ' Must specify at least one input.'
            )

        self._input_names_set = Set(initialize=self._input_names, ordered=True)

        if inputs is None:
            self.inputs = Var(self._input_names_set)
        else:
            if ex_model.n_inputs() != len(inputs):
                raise ValueError(
                    "Dimension mismatch in provided input vars for external "
                    "model.\nExpected %s input vars, got %s."
                    % (ex_model.n_inputs(), len(inputs))
                )
            self.inputs = Reference(inputs)

        self._equality_constraint_names = ex_model.equality_constraint_names()
        self._output_names = ex_model.output_names()

        self._output_names_set = Set(initialize=self._output_names, ordered=True)

        if outputs is None:
            self.outputs = Var(self._output_names_set)
        else:
            if ex_model.n_outputs() != len(outputs):
                raise ValueError(
                    "Dimension mismatch in provided output vars for external "
                    "model.\nExpected %s output vars, got %s."
                    % (ex_model.n_outputs(), len(outputs))
                )
            self.outputs = Reference(outputs)

        # call the callback so the model can set initialization, bounds, etc.
        external_grey_box_model.finalize_block_construction(self)

    def get_external_model(self):
        return self._ex_model


class ExternalGreyBoxBlock(Block):
    _ComponentDataClass = ExternalGreyBoxBlockData

    def __new__(cls, *args, **kwds):
        if cls != ExternalGreyBoxBlock:
            target_cls = cls
        elif not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            target_cls = ScalarExternalGreyBoxBlock
        else:
            target_cls = IndexedExternalGreyBoxBlock
        return super(ExternalGreyBoxBlock, cls).__new__(target_cls)

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', ExternalGreyBoxBlock)
        self._init_model = Initializer(kwds.pop('external_model', None))
        Block.__init__(self, *args, **kwds)

    def construct(self, data=None):
        """
        Construct the ExternalGreyBoxBlockDatas
        """
        if self._constructed:
            return
        # Do not set the constructed flag - Block.construct() will do that

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing external grey box model %s" % (self.name))

        super(ExternalGreyBoxBlock, self).construct(data)

        if self._init_model is not None:
            block = self.parent_block()
            for index, data in self.items():
                data.set_external_model(self._init_model(block, index))


class ScalarExternalGreyBoxBlock(ExternalGreyBoxBlockData, ExternalGreyBoxBlock):
    def __init__(self, *args, **kwds):
        ExternalGreyBoxBlockData.__init__(self, component=self)
        ExternalGreyBoxBlock.__init__(self, *args, **kwds)
        # The above inherit from Block and _BlockData, so it's not until here
        # that we know it's scalar. So we set the index accordingly.
        self._index = UnindexedComponent_index

    # Pick up the display() from Block and not BlockData
    display = ExternalGreyBoxBlock.display


class SimpleExternalGreyBoxBlock(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarExternalGreyBoxBlock
    __renamed__version__ = '6.0'


class IndexedExternalGreyBoxBlock(ExternalGreyBoxBlock):
    pass
