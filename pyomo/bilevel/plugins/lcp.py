#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

if False:
    # Delete these imports sometime soon
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

        This assumes that the original model has the form:

            min c1*x + d1*y
                x >= 0
                A1*x + B1*y <= b1
                min c2*x + d2*y
                    y >= 0
                    A2*x + B2*y <= b

        NOTE THE VARIABLE BOUNDS!
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
        # Collect objective info
        #
        d2 = {} 
        for (oname, odata) in submodel.active_components(Objective).items():
            for ndx in odata:
                if odata[ndx].sense == maximize:
                    raise IOError("Can only handle minimization submodels")
                o_terms = generate_canonical_repn(odata[ndx].expr, compute_values=False)
                for i in range(len(o_terms.variables)):
                    d2[ o_terms.variables[i].parent_component().name, o_terms.variables[i].index() ] = o_terms.linear[i]
            # Stop after the first objective
            break
        #
        # Collect constraints and setup the complementarity conditions
        #
        for (name, data) in block.active_components(Constraint).items():
            for ndx in data:
                con = data[ndx]
                body_terms = generate_canonical_repn(con.body, compute_values=False)
                lower_terms = generate_canonical_repn(con.lower, compute_values=False) if not con.lower is None else None
                upper_terms = generate_canonical_repn(con.upper, compute_values=False) if not con.upper is None else None
                #
                if body_terms.constant is None:
                    body_terms.constant = 0
                if not lower_terms is None and not lower_terms.variables is None:
                    raise(RuntimeError, "ERROR: Constraint '%s' has a lower bound that is non-constant")
                if not upper_terms is None and not upper_terms.variables is None:
                    raise(RuntimeError, "ERROR: Constraint '%s' has an upper bound that is non-constant")
                #
                for i in range(len(body_terms.variables)):
                    varname = body_terms.variables[i].parent_component().name
                    varndx = body_terms.variables[i].index()
                    A.setdefault(body_terms.variables[i].parent_component().name, {}).setdefault(varndx,[]).append( Bunch(coef=body_terms.linear[i], var=name, ndx=ndx) )
                #
                if not con.equality:
                    #
                    # Inequality constraint
                    #
                    if lower_terms is None or lower_terms.constant is None:
                        #
                        # body <= upper
                        #
                        v_domain[name, ndx] = -1
                        b2 = upper_terms.constant - body_terms.constant
                    elif upper_terms is None or upper_terms.constant is None:
                        #
                        # lower <= body
                        #
                        raise IOError("Cannot handle inequality constraints of the form l <= a^T x")
                    else:
                        #
                        # lower <= body <= upper
                        #
                        raise IOError("Cannot handle inequality constraints of the form l <= a^T x <= u")
                else:
                    #
                    # Equality constraint
                    #
                    raise IOError("Cannot handle inequality constraints of the form l = a^T")
        #
        # Return the block, which is added to the model
        #
        return block




def foo():
    #
    # Collect bound constraints
    #
    def all_vars(block):
        """
        This conditionally chains together the active variables in the current block with
        the active variables in all of the parent blocks (if any exist).
            """
        while not block is None:
            for (name, data) in block.active_components(Var).items():
                yield (name, data)
            block = block.parent_block()

    for (name, data) in all_vars(block):
        #
        # Skip fixed variables (in the parent)
        #
        if not (name, data.is_indexed()) in cnames:
            continue
        #
        # Iterate over all variable indices
        #
        for ndx in data:
            var = data[ndx]
            bounds = var.bounds
            if bounds[0] is None and bounds[1] is None:
                c_sense[name,ndx] = 'e'
            elif bounds[0] is None:
                if bounds[1] == 0.0:
                    c_sense[name,ndx] = 'g'
                else:
                    c_sense[name,ndx] = 'e'
                    #
                    # Add constraint that defines the upper bound
                    #
                    name_ = name + "_upper_"
                    varname = data.parent_component().name
                    varndx = data[ndx].index()
                    A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                    #
                    v_domain[name_,ndx] = -1
                    b_coef[name_,ndx] = bounds[1]
            elif bounds[1] is None:
                if bounds[0] == 0.0:
                    c_sense[name,ndx] = 'l'
                else:
                    c_sense[name,ndx] = 'e'
                    #
                    # Add constraint that defines the lower bound
                    #
                    name_ = name + "_lower_"
                    varname = data.parent_component().name
                    varndx = data[ndx].index()
                    A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                    #
                    v_domain[name_,ndx] = 1
                    b_coef[name_,ndx] = bounds[0]
            else:
                # Bounded above and below
                c_sense[name,ndx] = 'e'
                #
                # Add constraint that defines the upper bound
                #
                name_ = name + "_upper_"
                varname = data.parent_component().name
                varndx = data[ndx].index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = -1
                b_coef[name_,ndx] = bounds[1]
                #
                # Add constraint that defines the lower bound
                #
                name_ = name + "_lower_"
                varname = data.parent_component().name
                varndx = data[ndx].index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = 1
                b_coef[name_,ndx] = bounds[0]
    #
    return (A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain)

