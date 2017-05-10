#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import os

from pyomo.pysp.plugins.phhistoryextension import load_history

import matplotlib.pylab as plt

assert len(sys.argv) == 2
filename = sys.argv[1]
assert os.path.exists(filename)

scenario_tree, history, iterations = load_history(filename)

for node_name, node in scenario_tree['nodes'].items():

    # it's not a leaf node
    if len(node['children']):

        node_vars = history['0']['node solutions'][node_name]['variables'].keys()
        node_scenarios = node['scenarios']

        node_avg_res = {}
        node_xbar_res = {}
        scen_res = {}
        # will produce 1 figure for each variable name in this list
        VARS_TO_SHOW = node_vars
        for varname in VARS_TO_SHOW:
            node_avg_res[varname] = []
            node_xbar_res[varname] = []
            var_scen_res = scen_res[varname] = {}
            for scenario_name in node_scenarios:
                var_scen_res[scenario_name] = {'value':[],'weight':[]}

        for i in iterations:
            history_i = history[i]

            node_solution = history_i['node solutions'][node_name]['variables']
            for varname in VARS_TO_SHOW:
                node_avg_res[varname].append(node_solution[varname]['solution'])
                node_xbar_res[varname].append(node_solution[varname]['xbar'])
            del node_solution

            for scenario_name in node_scenarios:
                scenario_solution = history_i['scenario solutions'][scenario_name]['variables']
                for varname in VARS_TO_SHOW:
                    scen_res[varname][scenario_name]['value'].append(scenario_solution[varname]['value'])
                    scen_res[varname][scenario_name]['weight'].append(scenario_solution[varname]['weight'])
                del scenario_solution

            del history_i
        
        for varname in VARS_TO_SHOW:
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
