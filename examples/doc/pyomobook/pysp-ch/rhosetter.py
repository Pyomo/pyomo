#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#@pyomobook:
def ph_rhosetter_callback(ph, scenario_tree, scenario):
   
   MyRhoFactor = 1.0

   root_node = scenario_tree.findRootNode()

   si = scenario._instance
   sm = si._ScenarioTreeSymbolMap

   for i in si.ProductSizes:

      ph.setRhoOneScenario(
         root_node,
         scenario,
         sm.getSymbol(si.NumProducedFirstStage[i]),
         si.UnitProductionCosts[i] * MyRhoFactor * 0.001)

      for j in si.ProductSizes:
         if j <= i: 
            ph.setRhoOneScenario(
               root_node,
               scenario,
               sm.getSymbol(si.NumUnitsCutFirstStage[i,j]),
               si.UnitReductionCost * MyRhoFactor * 0.001)
#@:pyomobook
