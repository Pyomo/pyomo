def pysp_postinit_callback(scenario_tree_manager, scenario_tree, scenario):
    print("postinit callback1: "+str(scenario.name)+", "
          +str(scenario_tree_manager._aggregate_user_data['names']))
    print("")
    scenario_tree_manager._aggregate_user_data['names'].remove(scenario.name)
