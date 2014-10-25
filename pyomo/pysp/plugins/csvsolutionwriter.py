#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from pyomo.util.plugin import *
from pyomo.pysp import solutionwriter
from pyomo.pysp.scenariotree import *

from six import iteritems
#
# a simple utility to munge the index name into something a bit more csv-friendly and
# in general more readable. at the current time, we just eliminate any leading and trailing
# parentheses and change commas to colons - the latter because it's a csv file!
#

def index_to_string(index):

    result = str(index)
    result = result.lstrip('(').rstrip(')')
    result = result.replace(',',':')
    result = result.replace(' ','')

    return result

class CSVSolutionWriter(SingletonPlugin):

    implements (solutionwriter.ISolutionWriterExtension)

    def write(self, scenario_tree, output_file_prefix):

        if not isinstance(scenario_tree, ScenarioTree):
            raise RuntimeError("CSVSolutionWriter write method expects ScenarioTree object - type of supplied object="+str(type(scenario_tree)))

        output_filename = output_file_prefix + ".csv"
        output_file = open(output_filename,"w")

        for stage in scenario_tree._stages:
            stage_name = stage._name
            for tree_node in stage._tree_nodes:
                tree_node_name = tree_node._name
                for variable_id, (var_name, index) in iteritems(tree_node._variable_ids):
                    output_file.write(str(stage_name)+" , "+str(tree_node_name)+" , "+str(var_name)+" , "+str(index_to_string(index))+" , "+str(tree_node._solution[variable_id])+"\n")

                stage_cost_vardata = tree_node._cost_variable_datas[0][0]
                output_file.write(str(stage_name)+" , "+str(tree_node_name)+" , "+str(stage_cost_vardata.parent_component().name)+" , "+str(index_to_string(stage_cost_vardata.index()))+" , "+str(stage_cost_vardata())+"\n")

        output_file.close()



        print("Scenario tree solution written to file="+output_filename)

        # special, double-secret probationary code to write a file of cost variables so the user
        # can make sure that the cost variables are doing what they are supposed to be doing
         
        output_filename = "CostVarDetail" + ".csv"
        output_file = open(output_filename,"w")
        for stage in scenario_tree._stages:
            stage_name = stage._name
            for tree_node in stage._tree_nodes:
                for cost_var, scenprob in tree_node._cost_variable_datas:
                    output_file.write(str(stage_name)+" , "+str(tree_node._name)+" , "+str(cost_var.parent_component().name)+" , "+str(cost_var.parent_component().name)+" , "+str(index_to_string(cost_var.index()))+" , "+str(cost_var())+"\n")

        print(output_filename+" written for modeling checking.")
        output_file.close()
