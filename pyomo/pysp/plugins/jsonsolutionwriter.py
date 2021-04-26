#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.plugin import SingletonPlugin, implements
from pyomo.pysp import solutionwriter
from pyomo.pysp.scenariotree import ScenarioTree
from pyomo.pysp.plugins.phhistoryextension \
    import extract_scenario_tree_structure, \
           extract_scenario_solutions, \
           extract_node_solutions

import json

class JSONSolutionWriter(SingletonPlugin):

    implements (solutionwriter.ISolutionWriterExtension)

    def write(self, scenario_tree, output_file_prefix):

        if not isinstance(scenario_tree, ScenarioTree):
            raise RuntimeError("JSONSolutionWriter write method expects ScenarioTree object - type of supplied object="+str(type(scenario_tree)))

        include_ph_objective_parameters = None
        include_variable_statistics = None
        if output_file_prefix == 'ph':
            include_ph_objective_parameters = True
            include_variable_statistics = True
        elif output_file_prefix == 'postphef':
            include_ph_objective_parameters = False
            include_variable_statistics = True
        elif output_file_prefix == 'ef':
            include_ph_objective_parameters = False
            include_variable_statistics = False
        else:
            raise ValueError("JSONSolutionWriter requires an output prefix of 'ef', 'ph', or 'postphef' "
                             "to indicate whether ph specific parameter values should be extracted "
                             "from the solution")
        
        output_filename = output_file_prefix+"_solution.json"
        results = {}
        results['scenario tree'] = extract_scenario_tree_structure(scenario_tree)
        results['scenario solutions'] \
            = extract_scenario_solutions(scenario_tree,
                                         include_ph_objective_parameters=include_ph_objective_parameters)
        results['node solutions'] \
            = extract_node_solutions(scenario_tree,
                                     include_ph_objective_parameters=include_ph_objective_parameters,
                                     include_variable_statistics=include_variable_statistics)

        with open(output_filename,'w') as f:
            json.dump(results,f,indent=2)
        print("Scenario tree solution written to file="+output_filename)
