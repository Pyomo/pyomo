def pysp_postinit_callback(scenario_tree_manager, scenario_tree, scenario):
    print("postinit callback2: "+str(scenario._name)+", "
          +str(scenario_tree_manager._aggregate_user_data['probabilities']))
    print("")
