#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#

#@pyomobook:
# We only need to set upper bounds on first-stage variables, i.e., those
# being blended.

def ph_boundsetter_callback(ph, scenario_tree, scenario):

    # x is a first stage variable
    root_node = scenario_tree.findRootNode()
    scenario_instance = scenario._instance
    symbol_map = scenario_instance._ScenarioTreeSymbolMap
    for arc in scenario_instance.Arcs:
        scenario_instance.x[arc].setlb(0.0)
        scenario_instance.x[arc].setub(scenario_instance.M)
#@:pyomobook
