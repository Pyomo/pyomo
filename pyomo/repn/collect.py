#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# Routines to collect data in a structured format

from pyutilib.misc import Bunch
from pyomo.core.base import  Var, Constraint, Objective, maximize, minimize
from pyomo.repn import generate_canonical_repn


def collect_linear_terms(block, unfixed):
    #
    # Variables are constraints of block
    # Constraints are unfixed variables of block and the parent model.
    #
    vnames = set()
    for (name, data) in block.component_map(Constraint, active=True).items():
        vnames.add((name, data.is_indexed()))
    cnames = set(unfixed)
    for (name, data) in block.component_map(Var, active=True).items():
        cnames.add((name, data.is_indexed()))
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
    for (oname, odata) in block.component_map(Objective, active=True).items():
        for ndx in odata:
            if odata[ndx].sense == maximize:
                o_terms = generate_canonical_repn(-1*odata[ndx].expr, compute_values=False)
                d_sense = minimize
            else:
                o_terms = generate_canonical_repn(odata[ndx].expr, compute_values=False)
                d_sense = maximize
            for i in range(len(o_terms.variables)):
                c_rhs[ o_terms.variables[i].parent_component().local_name, o_terms.variables[i].index() ] = o_terms.linear[i]
        # Stop after the first objective
        break
    #
    # Collect constraints
    #
    for (name, data) in block.component_map(Constraint, active=True).items():
        for ndx in data:
            con = data[ndx]
            body_terms = generate_canonical_repn(con.body, compute_values=False)
            lower_terms = generate_canonical_repn(con.lower, compute_values=False) if not con.lower is None else None
            upper_terms = generate_canonical_repn(con.upper, compute_values=False) if not con.upper is None else None
            #
            if body_terms.constant is None:
                body_terms.constant = 0
            if not lower_terms is None and not lower_terms.variables is None:
                raise(RuntimeError, "Error during dualization:  Constraint '%s' has a lower bound that is non-constant")
            if not upper_terms is None and not upper_terms.variables is None:
                raise(RuntimeError, "Error during dualization:  Constraint '%s' has an upper bound that is non-constant")
            #
            for i in range(len(body_terms.variables)):
                varname = body_terms.variables[i].parent_component().local_name
                varndx = body_terms.variables[i].index()
                A.setdefault(body_terms.variables[i].parent_component().local_name, {}).setdefault(varndx,[]).append( Bunch(coef=body_terms.linear[i], var=name, ndx=ndx) )

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
                    b_coef[name,ndx] = upper_terms.constant - body_terms.constant
                elif upper_terms is None or upper_terms.constant is None:
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
    def all_vars(block):
        """
        This conditionally chains together the active variables in the current block with
        the active variables in all of the parent blocks (if any exist).
        """
        while not block is None:
            for (name, data) in block.component_map(Var, active=True).items():
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
                    varname = data.parent_component().local_name
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
                    varname = data.parent_component().local_name
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
                varname = data.parent_component().local_name
                varndx = data[ndx].index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = -1
                b_coef[name_,ndx] = bounds[1]
                #
                # Add constraint that defines the lower bound
                #
                name_ = name + "_lower_"
                varname = data.parent_component().local_name
                varndx = data[ndx].index()
                A.setdefault(varname, {}).setdefault(varndx,[]).append( Bunch(coef=1.0, var=name_, ndx=ndx) )
                #
                v_domain[name_,ndx] = 1
                b_coef[name_,ndx] = bounds[0]
    #
    return (A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain)
