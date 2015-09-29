def pysp_postinit_callback(worker, scenario):
    # test the order of callback execution
    assert scenario.name not in worker._aggregate_user_data['names']
    print("postinit callback2: "+str(scenario.name)+", "
          +str(worker._aggregate_user_data['leaf_node']))
    print("")
