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

def replicate_var(comp, name, block, index_set=None):
    """
    Create a new variable that will have the same indexing set, domain,
    and bounds as the provided component, and add it to the given block.
    Optionally pass an index set to use that to build the variable, but
    this set must be symmetric to comp's index set.
    """
    if index_set is None:
        if comp.is_indexed():
            index_set = comp.index_set()
        else:
            index_set = UnindexedComponent_set

    var_args = {}
    try:
        var_args['domain'] = comp.domain
    except AttributeError:
        pass
    try:
        var_args['bounds'] = comp.bounds
    except AttributeError:
        pass

    new_var = Var(index_set, **var_args)
    block.add_component(name, new_var)
    if comp.is_indexed():
        for i in index_set:
            try:
                # set bounds for every member in case they differ
                new_var[i].domain = comp[i].domain
                new_var[i].setlb(comp[i].lb)
                new_var[i].setub(comp[i].ub)
            except AttributeError:
                break

    return new_var
