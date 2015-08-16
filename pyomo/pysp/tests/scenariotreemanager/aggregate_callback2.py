def pysp_aggregategetter_callback(scenario_tree_manager,
                                  scenario_tree,
                                  scenario,
                                  data):

    data.setdefault('probabilities',[]).append(scenario._probability)
    print("aggregategetter callback2: "+str(scenario._name)+", "
          +str(data))
    print("")

    # **IMPT**: Must also return aggregate data in a singleton tuple
    #           to work with bundles
    return (data,)
