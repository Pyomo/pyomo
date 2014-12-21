#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *

# This callback is for collecting aggregate scenario data which is
# stored on the _aggregate_user_data member of ph after this callback
# has been sequentially executed with every scenario.  This is the
# only reliable method for collecting such data because scenario
# instances are not present on the master ph object when PH is
# executed in parallel mode.

def ph_aggregategetter_callback(ph, scenario_tree, scenario, data):

    if 'max_aggregate_demand' not in data:
        data['max_aggregate_demand'] = 0.0

    # a super-weak upper bound to be used with the boundsetter callback,
    # namely the max aggregate demand observed in any scenario
    instance = scenario._instance
    scenario_aggregate_demand = sum([value(instance.Demand[i,j]) \
                                     for (i,j) in instance.Arcs])
    if scenario_aggregate_demand > data['max_aggregate_demand']:
        data['max_aggregate_demand'] = scenario_aggregate_demand

    # **IMPT**: Must also return aggregate data in a singleton tuple
    #           to work with bundles
    return (data,)
