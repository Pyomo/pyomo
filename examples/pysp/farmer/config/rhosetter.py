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

   scenario_yield = ph._aggregate_user_data['scenario_yield']
   max_yield = ph._aggregate_user_data['max_yield']
   min_yield = ph._aggregate_user_data['min_yield']

   scenario_instance = scenario._instance
   for c in scenario_instance.CROPS:
      assert min_yield[c] <= scenario_yield[scenario._name][c] <= max_yield[c]

   symbol_map = scenario_instance._ScenarioTreeSymbolMap
   for c in scenario_instance.CROPS:
      variable_id = symbol_map.getSymbol(scenario_instance.DevotedAcreage[c])
      ph.setRhoOneScenario(root_node,
                           scenario,
                           variable_id,
                           1.0/max_yield[c])
