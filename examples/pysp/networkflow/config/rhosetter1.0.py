
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
