#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Routines to collect data in a structured format

from pyomo.common.collections import Bunch
from pyomo.core.base import  Var, Constraint, Objective, maximize, minimize
from pyomo.repn.standard_repn import generate_standard_repn


def collect_linear_terms(block, unfixed):
    #
    # Variables are constraints of block
    # Constraints are unfixed variables of block and the parent model.
    #
    vnames = set()
    for obj in block.component_objects(Constraint, active=True):
        vnames.add((obj.getname(fully_qualified=True, relative_to=block), obj.is_indexed()))
    cnames = set(unfixed)
    for obj in block.component_objects(Var, active=True):
        cnames.add((obj.getname(fully_qualified=True, relative_to=block), obj.is_indexed()))
    #
    A = {}
    b_coef = {}
    c_rhs = {}
    c_sense = {}
    d_sense = None
    v_domain = {}
    #
    # Collect objective
    #
    for odata in block.component_objects(Objective, active=True):
        for ndx in odata:
            if odata[ndx].sense == maximize:
                o_terms = generate_standard_repn(-1*odata[ndx].expr, compute_values=False)
                d_sense = minimize
            else:
                o_terms = generate_standard_repn(odata[ndx].expr, compute_values=False)
                d_sense = maximize
            for var, coef in zip(o_terms.linear_vars, o_terms.linear_coefs):
                c_rhs[ var.parent_component().local_name, var.index() ] = coef
        # Stop after the first objective
        break
    #
    # Collect constraints
    #
    for data in block.component_objects(Constraint, active=True):
        name = data.getname(relative_to=block)
        for ndx in data:
            con = data[ndx]
            body_terms = generate_standard_repn(con.body, compute_values=False)
            if body_terms.is_fixed():
                #
                # If a constraint has a fixed body, then don't collect it.
                #
                continue
            lower_terms = generate_standard_repn(con.lower, compute_values=False) if not con.lower is None else None
            upper_terms = generate_standard_repn(con.upper, compute_values=False) if not con.upper is None else None
            #
            if not lower_terms is None and not lower_terms.is_constant():
                raise(RuntimeError, "Error during dualization:  Constraint '%s' has a lower bound that is non-constant")
            if not upper_terms is None and not upper_terms.is_constant():
                raise(RuntimeError, "Error during dualization:  Constraint '%s' has an upper bound that is non-constant")
            #
            for var, coef in zip(body_terms.linear_vars, body_terms.linear_coefs):
                try:
                    # The variable is in the subproblem
                    varname = var.parent_component().getname(fully_qualified=True, relative_to=block)
                except:
                    # The variable is somewhere else in the model
                    varname = var.parent_component().getname(fully_qualified=True, relative_to=block.model())
                varndx = var.index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=coef, var=name, ndx=ndx) )
            #
            if not con.equality:
                #
                # Inequality constraint
                #
                if lower_terms is None:
                    #
                    # body <= upper
                    #
                    v_domain[name, ndx] = -1
                    b_coef[name,ndx] = upper_terms.constant - body_terms.constant
                elif upper_terms is None:
                    #
                    # lower <= body
                    #
                    v_domain[name, ndx] = 1
                    b_coef[name,ndx] = lower_terms.constant - body_terms.constant
                else:
                    #
                    # lower <= body <= upper
                    #
                    # Dual for lower bound
                    #
                    ndx_ = tuple(list(ndx).append('lb'))
                    v_domain[name, ndx_] = 1
                    b_coef[name,ndx] = lower_terms.constant - body_terms.constant
                    #
                    # Dual for upper bound
                    #
                    ndx_ = tuple(list(ndx).append('ub'))
                    v_domain[name, ndx_] = -1
                    b_coef[name,ndx] = upper_terms.constant - body_terms.constant
            else:
                #
                # Equality constraint
                #
                v_domain[name, ndx] = 0
                b_coef[name,ndx] = lower_terms.constant - body_terms.constant
    #
    # Collect bound constraints
    #
    def all_vars(b):
        """
        This conditionally chains together the active variables in the current block with
        the active variables in all of the parent blocks (if any exist).
        """
        for obj in b.component_objects(Var, active=True, descend_into=True):
            name = obj.parent_component().getname(fully_qualified=True, relative_to=b)
            yield (name, obj)
        #
        # Look through parent blocks
        #
        b = b.parent_block()
        while not b is None:
            for obj in b.component_objects(Var, active=True, descend_into=False):
                name = obj.parent_component().name
                yield (name, obj)
            b = b.parent_block()

    for name, data in all_vars(block):
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
                    varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
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
                    varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
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
                varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
                varndx = data[ndx].index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = -1
                b_coef[name_,ndx] = bounds[1]
                #
                # Add constraint that defines the lower bound
                #
                name_ = name + "_lower_"
                varname = data.parent_component().getname(fully_qualified=True, relative_to=block)
                varndx = data[ndx].index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = 1
                b_coef[name_,ndx] = bounds[0]
    #
    return (A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain)
