# This callback must be used with the aggregategetter.py
# callback in order to populate the 
# _aggregate_user_data object.

# only need to set upper bounds on first-stage variables, i.e., those
# being blended.

def ph_boundsetter_callback(ph, scenario_tree, scenario):

    # x is a first stage variable
    root_node = scenario_tree.findRootNode()

    scenario_instance = scenario._instance

    symbol_map = scenario_instance._ScenarioTreeSymbolMap

    max_aggregate_demand = ph._aggregate_user_data['max_aggregate_demand']

    for arc in scenario_instance.Arcs:

        scenario_instance.x[arc].setlb(0.0)
        scenario_instance.x[arc].setub(max_aggregate_demand)

