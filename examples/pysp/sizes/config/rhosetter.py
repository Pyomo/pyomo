
def ph_rhosetter_callback(ph, scenario_tree, scenario):
   
   MyRhoFactor = 1.0

   root_node = scenario_tree.findRootNode()

   scenario_instance = scenario._instance
   symbol_map = scenario_instance._ScenarioTreeSymbolMap

   for i in scenario_instance.ProductSizes:

      ph.setRhoOneScenario(
         root_node,
         scenario,
         symbol_map.getSymbol(scenario_instance.ProduceSizeFirstStage[i]),
         scenario_instance.SetupCosts[i] * MyRhoFactor * 0.001)

      ph.setRhoOneScenario(
         root_node,
         scenario,
         symbol_map.getSymbol(scenario_instance.NumProducedFirstStage[i]),
         scenario_instance.UnitProductionCosts[i] * MyRhoFactor * 0.001)

      for j in scenario_instance.ProductSizes:
         if j <= i: 
            ph.setRhoOneScenario(
               root_node,
               scenario,
               symbol_map.getSymbol(scenario_instance.NumUnitsCutFirstStage[i,j]),
               scenario_instance.UnitReductionCost * MyRhoFactor * 0.001)
