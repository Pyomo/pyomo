def pysp_postinit_callback(worker, scenario):
    print("postinit callback1: "+str(scenario.name)+", "
          +str(worker._aggregate_user_data['names']))
    print("")
    worker._aggregate_user_data['names'].remove(scenario.name)
