#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import *

# This callback is for collecting aggregate scenario data which is
# stored on the _aggregate_user_data member of ph after this callback
# has been sequentially executed with every scenario.  This is the
# only reliable method for collecting such data because scenario
# instances are not present on the master ph object when PH is
# executed in parallel mode.

def ph_aggregategetter_callback(ph, scenario_tree, scenario, data):

    if 'scenario_yield' not in data:
        # This is the first time calling
        scenario_yield = data['scenario_yield'] = {}
        max_yield = data['max_yield'] = {}
        max_yield['WHEAT'] = 0.0
        max_yield['CORN'] = 0.0
        max_yield['SUGAR_BEETS'] = 0.0
        min_yield = data['min_yield'] = {}
        min_yield['WHEAT'] = float('Inf')
        min_yield['CORN'] = float('Inf')
        min_yield['SUGAR_BEETS'] = float('Inf')
    else:
        scenario_yield = data['scenario_yield']
        max_yield = data['max_yield']
        min_yield = data['min_yield']

    this_scenario_yield = scenario_yield[scenario._name] = {}
    max_yield = max_yield
    min_yield = min_yield
    instance = scenario._instance
    for c in instance.CROPS:
        crop_yield = this_scenario_yield[c] = value(instance.Yield[c])
        if crop_yield > max_yield[c]:
            max_yield[c] = crop_yield
        if crop_yield < min_yield[c]:
            min_yield[c] = crop_yield

    # **IMPT**: Must also return aggregate data in a singleton tuple
    #           to work with bundles
    return (data,)
