#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import *

def ph_boundsetter_callback(ph, scenario_tree, scenario):

   scenario_yield = ph._aggregate_user_data['scenario_yield']
   max_yield = ph._aggregate_user_data['max_yield']
   min_yield = ph._aggregate_user_data['min_yield']

   scenario_instance = scenario._instance
   for c in scenario_instance.CROPS:
      assert min_yield[c] <= scenario_yield[scenario._name][c] <= max_yield[c]

   symbol_map = scenario_instance._ScenarioTreeSymbolMap
   leaf_node = scenario._node_list[-1]
   for c in scenario_instance.CROPS:

      max_produced = \
         max_yield[c]*value(scenario_instance.TOTAL_ACREAGE)

      variable_id = \
         symbol_map.getSymbol(scenario_instance.QuantitySuperQuotaSold[c])
      ph.setVariableBoundsOneScenario(
         leaf_node,
         scenario,
         variable_id,
         0.0,
         max_produced)

      variable_id = \
         symbol_map.getSymbol(scenario_instance.QuantitySubQuotaSold[c])
      ph.setVariableBoundsOneScenario(
         leaf_node,
         scenario,
         variable_id,
         0.0,
         max_produced)
