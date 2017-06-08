def pysp_aggregategetter_callback(worker,
                                  scenario,
                                  data):

    data.setdefault('names',[]).append(scenario.name)
    print("aggregategetter callback1: "+str(scenario.name)+", "
          +str(data))
    print("")

    # **IMPT**: Must also return aggregate data in a singleton tuple
    #           to work with bundles
    return (data,)
