def pysp_postinit_callback(scenario_tree_manager, scenario_tree, scenario):
    # test the order of callback execution
    assert scenario.name not in scenario_tree_manager._aggregate_user_data['names']
    print("postinit callback2: "+str(scenario.name)+", "
          +str(scenario_tree_manager._aggregate_user_data['leaf_node']))
    print("")
