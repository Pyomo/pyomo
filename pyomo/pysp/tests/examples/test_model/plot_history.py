#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import matplotlib.pylab as plt
import json
import shelve
import sys
import os

assert len(sys.argv) == 2
filename = sys.argv[1]
assert os.path.exists(filename)

history = None
try:
    with open(filename) as f:
        history = json.load(f)
except:
    history = None
    try:
        history = shelve.open(filename,
                              flag='r')
    except:
        history = None

if history is None:
    raise RuntimeError("Unable to open ph history file as JSON "
                           "or python Shelve DB format")

scenario_tree = history['scenario tree']

try:
    iter_keys = history['results keys']
except KeyError:
    # we are using json format (which loads the entire file anyway)
    iter_keys = list(history.keys())
    iter_keys.remove('scenario tree')
iterations = sorted(int(k) for k in iter_keys)
iterations = [str(k) for k in iterations]

for node_name, node in scenario_tree['nodes'].items():

    # it's not a leaf node
    if len(node['children']):

        node_vars = history['0']['node solutions'][node_name]['variables'].keys()
        node_scenarios = node['scenarios']

        node_avg_res = {}
        node_xbar_res = {}
        scen_res = {}
        for varname in node_vars:
            node_avg_res[varname] = []
            node_xbar_res[varname] = []
            var_scen_res = scen_res[varname] = {}
            for scenario_name in node_scenarios:
                var_scen_res[scenario_name] = {'value':[],'weight':[]}

        for i in iterations:
            history_i = history[i]

            node_solution = history_i['node solutions'][node_name]['variables']
            for varname in node_vars:
                node_avg_res[varname].append(node_solution[varname]['solution'])
                node_xbar_res[varname].append(node_solution[varname]['xbar'])
            del node_solution

            for scenario_name in node_scenarios:
                scenario_solution = history_i['scenario solutions'][scenario_name]['variables']
                for varname in node_vars:
                    scen_res[varname][scenario_name]['value'].append(scenario_solution[varname]['value'])
                    scen_res[varname][scenario_name]['weight'].append(scenario_solution[varname]['weight'])
                del scenario_solution

            del history_i
        
        for varname in node_vars:
            figure = plt.figure()
            ax = figure.add_subplot(121)
            for scenario_name in node_scenarios:
                ax.plot(scen_res[varname][scenario_name]['value'],label=scenario_name)
            ax.plot(node_avg_res[varname],'k--',label='Node Average')
            ax.plot(node_xbar_res[varname],'k--',label='Node Xbar')
            ax.set_title(node_name+' - '+varname)
            if len(node_scenarios) <= 4:
                ax.legend(loc=0)

            ax = figure.add_subplot(122)
            for scenario_name in node_scenarios:
                ax.plot(scen_res[varname][scenario_name]['weight'],label=scenario_name)
            ax.set_title(node_name+' - '+varname)
            if len(node_scenarios) <= 4:
                ax.legend(loc=0)

plt.show()
