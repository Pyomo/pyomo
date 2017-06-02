#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import Param, value

def create_expected_value_instance(average_instance,
                                   scenario_tree,
                                   scenario_instances,
                                   verbose=False):

    rootnode = scenario_tree._stages[0]._tree_nodes[0]
    ScenCnt = len(rootnode._scenarios)

    for p in average_instance.component_map(Param, active=True):

        average_parameter_object = getattr(average_instance, p)

        for index in average_parameter_object:
            average_value = 0.0
            for scenario in rootnode._scenarios:
                scenario_parameter_object = getattr(scenario_instances[scenario._name], p)
                average_value += value(scenario_parameter_object[index])
            average_value = average_value / float(len(scenario_instances))
            average_parameter_object[index] = average_value

def fix_ef_first_stage_variables(ph, scenario_tree, expected_value_instance):

    if ph._verbose:
        print("Fixing first stage variables at mean instance solution values.\n")

    stage = ph._scenario_tree._stages[0]
    root_node = stage._tree_nodes[0] # there should be only one root node!
    for variable_name, index_template in stage._variable_templates.iteritems():

        variable_indices = root_node._variable_indices[variable_name]
        for index in variable_indices:
            for scen in root_node._scenarios:
                inst = ph._instances[scen._name]
                print("HEYYYY fix varstatus !!!!!xxxxxx\n")
                #if getattr(inst, variable_name)[index].status != VarStatus.unused:
                if 1 == 1:
                    print("variable_name= %s\n" % variable_name)
                    fix_value = getattr(expected_value_instance, variable_name)[index].value
                    getattr(inst, variable_name)[index].fix(fix_value)
