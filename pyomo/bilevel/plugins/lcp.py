#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

if False:
    from pyomo.core.base import Transformation, Var, Constraint, VarList, ConstraintList, Objective, Set, maximize, minimize, NonNegativeReals, NonPositiveReals, Reals, Block, ComponentUID
    from pyomo.repn.canonical_repn import generate_canonical_repn
    from pyomo.repn.canonical_repn import LinearCanonicalRepn
    from pyomo.core.plugins.transform.util import process_canonical_repn
    from pyomo.bilevel import SubModel
    from pyomo.core.base.plugin import TransformationFactory

from pyomo.repn.collect import collect_linear_terms
from pyomo.util.plugin import alias
from pyomo.bilevel.plugins.transform import Base_BilevelTransformation


import logging
logger = logging.getLogger('pyomo.core')


class LinearComplementarity_BilevelTransformation(Base_BilevelTransformation):

    alias('bilevel.linear_mpec', doc="Generate a linear MPEC from the optimality conditions of the submodel")

    def __init__(self):
        super(LinearComplementarity_BilevelTransformation, self).__init__()

    def apply(self, instance, **kwds):
        #
        # Process options
        #
        submodel = self._preprocess(instance, **kwds)
        self._fix_all()
        #
        # Create a block with optimality conditions
        #
        setattr(instance, self._submodel+'_kkt', self._add_optimality_conditions(submodel))
        #-------------------------------------------------------------------------------
        #
        # Deactivate the original subproblem
        #
        submodel.deactivate()
        #
        # Unfix the upper variables
        #
        self._fix_all()
        #
        # Disable the original submodel and
        # execute the preprocessor
        #
        submodel.deactivate()
        instance.preprocess()
        return instance

    def _add_optimality_conditions(self, submodel):
        """
        Add optimality conditions for the submodel
        """ 
        A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain = collect_linear_terms(block, self._unfixed_upper_vars)
        #
        # Populate the block with the linear constraints.  Note that we don't simply clone the
        # current block.  We need to collect a single set of equations that can be easily 
        # expressed.
        #
        block = Block()
        block.x = VarList()
        block.c = ConstraintList()
        #
        # Add the linear constraints
        #
        
        #
        return block

