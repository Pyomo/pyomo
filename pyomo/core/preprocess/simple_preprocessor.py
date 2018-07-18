#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.misc
import pyomo.common.plugin
import pyomo.common
from pyomo.core.base import IPyomoPresolveAction


@pyomo.common.pyomo_api(namespace='pyomo.model')
def simple_preprocessor(data, model=None):
    """
    This plugin simply applies preprocess actions in a fixed order.

    Required:
        model:      A concrete model instance.
    """
    pyomo.common.PyomoAPIFactory('pyomo.repn.compute_standard_repn')(data, model=model)
    #
    # Process the presolver actions
    #
    actions = pyomo.common.plugin.ExtensionPoint(IPyomoPresolveAction)
    active_actions = set()
    action_rank = {}
    #
    # Collect active actions
    #
    for action in actions():
        tmp = actions.service(action)
        if tmp is None:
            raise ValueError("Cannot activate unknown action %s" % action)
        active_actions.add(action)
        if rank is None:
            rank = tmp.rank()
        action_rank[action] = rank
    #
    # Sort active actions
    #
    actions = list(active_actions)
    ranks = []
    for item in actions:
        ranks.append(action_rank[item])
    index = pyutilib.misc.sort_index(ranks)
    sorted=[]
    for i in index:
        sorted.append( actions[i] )
    #
    # Preprocess active actions in order
    #
    for action in sorted:
        model = actions.service(action).preprocess(model)
