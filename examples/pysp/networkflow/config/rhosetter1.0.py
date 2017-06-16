#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
