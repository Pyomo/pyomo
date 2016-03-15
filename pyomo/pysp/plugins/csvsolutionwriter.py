#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.util.plugin
from pyomo.pysp import solutionwriter
from pyomo.pysp.scenariotree.tree_structure import \
    ScenarioTree

#
# a simple utility to munge the index name into something a
# bit more csv-friendly and in general more readable. at the
# current time, we just eliminate any leading and trailing
# parentheses and change commas to colons - the latter
# because it's a csv file!
#

def index_to_string(index):

    result = str(index)
    result = result.lstrip('(').rstrip(')')
    result = result.replace(',',':')
    result = result.replace(' ','')

    return result

class CSVSolutionWriter(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements(
        solutionwriter.ISolutionWriterExtension)

    def write(self, scenario_tree, output_file_prefix):

        if not isinstance(scenario_tree, ScenarioTree):
            raise RuntimeError(
                "CSVSolutionWriter write method expects "
                "ScenarioTree object - type of supplied "
                "object="+str(type(scenario_tree)))

        solution_filename = output_file_prefix + ".csv"
        with open(solution_filename, "w") as f:
            for stage in scenario_tree.stages:
                for tree_node in sorted(stage.nodes,
                                        key=lambda x: x.name):
                    for variable_id in sorted(tree_node._variable_ids):
                        var_name, index = \
                            tree_node._variable_ids[variable_id]
                        f.write("%s, %s, %s, %s, %s\n"
                                % (stage.name,
                                   tree_node.name,
                                   var_name,
                                   index_to_string(index),
                                   tree_node._solution[variable_id]))

                # GH: I am removing this code because
                #     it assumes convergence and stage costs
                #     are written in the CostVarDetail.csv file
                #     in the next loop anyway.
                #stage_cost_vardata = tree_node._cost_variable_datas[0][0]
                #f.write(str(stage.name)+" , "+
                #        str(tree_node.name)+" , "+
                #        str(stage._cost_variable[0])+" , "+
                #        str(index_to_string(stage._cost_variable[1]))+" , "+
                #        str(stage_cost)+"\n")
        print("Scenario tree solution written to file="+solution_filename)

        cost_filename = output_file_prefix + "_StageCostDetail.csv"
        with open(cost_filename, "w") as f:
            for stage in scenario_tree.stages:
                cost_name, cost_index = stage._cost_variable
                for tree_node in sorted(stage.nodes,
                                        key=lambda x: x.name):
                    for scenario in sorted(tree_node.scenarios,
                                           key=lambda x: x.name):
                        stage_cost = scenario._stage_costs[stage.name]
                        f.write("%s, %s, %s, %s, %s, %s\n"
                                % (stage.name,
                                   tree_node.name,
                                   scenario.name,
                                   cost_name,
                                   index_to_string(cost_index),
                                   stage_cost))
        print("Scenario stage costs written to file="+cost_filename)
