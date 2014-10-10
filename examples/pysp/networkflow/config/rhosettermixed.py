
def ph_rhosetter_callback(ph, scenario_tree, scenario):

   root_node = scenario_tree.findRootNode()

   scenario_instance = scenario._instance

   symbol_map = scenario_instance._ScenarioTreeSymbolMap

   for a in scenario_instance.Arcs:
      
      ph.setRhoOneScenario(root_node,
                           scenario,
                           symbol_map.getSymbol(scenario_instance.b0[a]),
                           scenario_instance.b0Cost[a] * 1.0)
      ph.setRhoOneScenario(root_node,
                           scenario,
                           symbol_map.getSymbol(scenario_instance.x[a]),
                           scenario_instance.CapCost[a] * 0.1)

       
