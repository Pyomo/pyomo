#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import Var
from pyomo.core.base.indexed_component import UnindexedComponent_set

def create_var(comp, name, block, index_set=None):
    if index_set is None:
        if comp.is_indexed():
            index_set = comp.index_set()
        else:
            index_set = UnindexedComponent_set

    new_var = Var(index_set)
    block.add_component(name, new_var)
    return new_var

def _tighten(src, dest):
    starting_lb = dest.lb
    starting_ub = dest.ub
    if not src.is_continuous():
        dest.domain = src.domain
    if src.lb is not None:
        if starting_lb is None:
            dest.setlb(src.lb)
        else:
            dest.setlb(max(starting_lb, src.lb))
    if src.ub is not None:
        if starting_ub is None:
            dest.setub(src.ub)
        else:
            dest.setub(min(starting_ub, src.ub))

def tighten_var_domain(comp, new_var, index_set=None):
    if index_set is None:
        if comp.is_indexed():
            index_set = comp.index_set()
        else:
            index_set = UnindexedComponent_set

    if comp.is_indexed():
        for i in index_set:
            try:
                # set bounds for every member in case they differ
                _tighten(comp[i], new_var[i])
            except AttributeError:
                break
    else:
        try:
            # set bounds for every member in case they differ
            _tighten(comp, new_var)
        except AttributeError:
            pass

    return new_var

def replicate_var(comp, name, block, index_set=None):
    """
    Create a new variable that will have the same indexing set, domain,
    and bounds as the provided component, and add it to the given block.
    Optionally pass an index set to use that to build the variable, but
    this set must be symmetric to comp's index set.
    """
    new_var = create_var(comp, name, block, index_set)
    tighten_var_domain(comp, new_var, index_set)
    return new_var
