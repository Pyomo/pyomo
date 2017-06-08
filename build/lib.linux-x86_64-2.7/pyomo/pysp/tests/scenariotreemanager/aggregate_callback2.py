def pysp_aggregategetter_callback(worker,
                                  scenario,
                                  data):
    assert 'names' in data
    data.setdefault('leaf_node',[]).append(scenario.leaf_node.name)
    print("aggregategetter callback2: "+str(scenario.name)+", "
          +str(data))
    print("")

    # **IMPT**: Must also return aggregate data in a singleton tuple
    #           to work with bundles
    return (data,)
