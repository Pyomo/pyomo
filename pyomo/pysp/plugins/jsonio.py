#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('JSONSolutionIOExtension',)

import itertools
import json

from pyomo.pysp.solutionioextensions import \
    (IPySPSolutionSaverExtension,
     IPySPSolutionLoaderExtension)
from pyomo.util.plugin import implements, SingletonPlugin
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_register_common_option)
from pyomo.pysp.util.configured_object import (PySPConfiguredObject,
                                               PySPConfiguredExtension)


def extract_node_solution(tree_node):
    solution = {}
    for variable_id in tree_node._standard_variable_ids:
        varname, index = tree_node._variable_ids[variable_id]
        if varname not in solution:
            solution[varname] = {}
        solution[varname][index] = tree_node._solution[variable_id]
    return solution

def load_node_solution(tree_node, solution):
    for varname in solution:
        varsol = solution[varname]
        for index in varsol:
            variable_id = tree_node._name_index_to_id[(varname, index)]
            tree_node._solution[variable_id] = varsol[index]

class JSONSolutionIOExtension(PySPConfiguredExtension,
                              PySPConfiguredObject,
                              SingletonPlugin):

    implements(IPySPSolutionSaverExtension)
    implements(IPySPSolutionLoaderExtension)

    _registered_options = \
        PySPConfigBlock("Options registered for the "
                        "JSONSolutionIOExtension class")

    safe_register_common_option(_registered_options,
                                "output_name")
    safe_register_common_option(_registered_options,
                                "input_name")
    safe_register_common_option(_registered_options,
                                "store_stages")
    safe_register_common_option(_registered_options,
                                "load_stages")

    _default_prefix = "jsonio_"

    #
    # Note: Do not try to user super() or access the
    #       class name inside the __init__ method when
    #       a class derives from a SingletonPlugin. Due to
    #       how Pyutilib implements its Singleton type,
    #       the __class__ cell will be empty.
    #       (See: https://stackoverflow.com/questions/
    #             13126727/how-is-super-in-python-3-implemented)
    #
    def __init__(self):
        PySPConfiguredExtension.__init__(self)

    def load(self, manager):

        if self.get_option("input_name") is not None:
            stage_solutions = None
            with open(self.get_option("input_name"), 'rb') as f:
                stage_solutions = json.load(f)
            for cnt, (stage, stage_solution) in \
                enumerate(itertools.izip_longest(manager.scenario_tree.stages,
                                                 stage_solutions), 1):
                if stage_solution is None:
                    break
                if (self.get_option('store_stages') <= 0) or \
                   (cnt <= self.get_option('store_stages')):
                    if stage is None:
                        raise RuntimeError(
                            "Local scenario tree has fewer stages than what is "
                            "held by the solution loaded from file %s. Use the "
                            "%s option 'load_stages' to limit the number of "
                            "stages that are loaded.")
                    for tree_node in stage.nodes:
                        try:
                            node_solution = stage_solution[tree_node._name]
                        except KeyError:
                            raise KeyError("Local scenario tree contains a tree node "
                                           "that was not found in the solution at time"
                                           "-stage %s: %s" % (cnt, tree_node._name))
                        load_node_solution(tree_node, node_solution)
                else:
                    break
                print("Loaded scenario tree solution for %s time stages" % (cnt))
            return True

        print("No value was set for %s option 'input_name'. "
              "Nothing will be saved." % (type(self).__name__))
        return False

    def save(self, manager):

        if self.get_option("output_name") is not None:
            stage_solutions = []
            for cnt, stage in enumerate(manager.scenario_tree.stages, 1):
                if (self.get_option('load_stages') <= 0) or \
                   (cnt <= self.get_option('load_stages')):
                    node_solutions = {}
                    for tree_node in stage.nodes:
                        node_solutions[tree_node.name] = \
                            extract_node_solution(tree_node)
                    stage_solutions.append(node_solutions)
                else:
                    break
            with open(self.get_option('output_name'), 'wb') as f:
                json.dump(stage_solutions, f, indent=2)
            return True

        print("No value was set for %s option 'output_name'. "
              "Nothing will be saved." % (type(self).__name__))
        return False
