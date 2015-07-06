#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

def ph_rhosetter_callback(ph, scenario_tree, scenario):

   root_node = scenario_tree.findRootNode()

   scenario_instance = scenario._instance

   symbol_map = scenario_instance._ScenarioTreeSymbolMap
   
   MyRhoFactor = 1.0

   for a in scenario_instance.Arcs:
      
      ph.setRhoOneScenario(root_node,
                           scenario,
                           symbol_map.getSymbol(scenario_instance.b0[a]),
                           scenario_instance.b0Cost[a] * MyRhoFactor)
      ph.setRhoOneScenario(root_node,
                           scenario,
                           symbol_map.getSymbol(scenario_instance.x[a]),
                           scenario_instance.CapCost[a] * MyRhoFactor)
