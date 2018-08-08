#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ()

import json

from pyomo.pysp.solutionioextensions import \
    (IPySPSolutionSaverExtension,
     IPySPSolutionLoaderExtension)
from pyomo.pysp.phutils import indexToString
from pyomo.common.plugin import implements, SingletonPlugin
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.util.configured_object import (PySPConfiguredObject,
                                               PySPConfiguredExtension)

try:
    from six.moves import zip_longest
except:
    zip_longest = None

def load_node_solution(tree_node, solution):
    for varname in solution:
        varsol = solution[varname]
        for index, val in varsol:
            if type(index) is list:
                variable_id = \
                    tree_node._name_index_to_id[(varname, tuple(index))]
            else:
                variable_id = tree_node._name_index_to_id[(varname, index)]
            tree_node._solution[variable_id] = val

class JSONSolutionLoaderExtension(PySPConfiguredExtension,
                                  PySPConfiguredObject,
                                  SingletonPlugin):

    implements(IPySPSolutionLoaderExtension)

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "input_name")
        safe_declare_common_option(options,
                                   "load_stages")

        return options

    _default_options_prefix = "jsonloader_"

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
            # Do NOT open file in 'binary' mode when loading JSON
            # (produces an error in Python3)
            with open(self.get_option("input_name"), 'r') as f:
                stage_solutions = json.load(f)
            cntr = 0
            if self.get_option('load_stages') > len(manager.scenario_tree.stages):
                raise ValueError("The value of the %s option (%s) can not be greater than "
                                 "the number of time stages in the local scenario tree (%s)"
                                 % (self.get_full_option_name('load_stages'),
                                    self.get_option('load_stages'),
                                    len(manager.scenario_tree.stages)))
            if self.get_option('load_stages') > len(stage_solutions):
                raise ValueError("The value of the %s option (%s) can not be greater than "
                                 "the number of time stages in the scenario tree solution "
                                 "stored in %s (%s)"
                                 % (self.get_full_option_name('load_stages'),
                                    self.get_option('load_stages'),
                                    self.get_option('input_name'),
                                    len(stage_solutions)))
            for stage, stage_solution in zip_longest(manager.scenario_tree.stages,
                                                     stage_solutions):
                if stage_solution is None:
                    break
                if (self.get_option('load_stages') <= 0) or \
                   (cntr+1 <= self.get_option('load_stages')):
                    if stage is None:
                        raise RuntimeError(
                            "Local scenario tree has fewer stages (%s) than what is "
                            "held by the solution loaded from file %s. Use the "
                            "option %s to limit the number of stages that "
                            "are loaded." % (cntr,
                                             self.get_option('input_name'),
                                             self.get_full_option_name('load_stages')))
                    cntr += 1
                    for tree_node in stage.nodes:
                        try:
                            node_solution = stage_solution[tree_node.name]
                        except KeyError:
                            raise KeyError("Local scenario tree contains a tree node "
                                           "that was not found in the solution at time"
                                           "-stage %s: %s" % (cntr, tree_node.name))
                        load_node_solution(tree_node, node_solution)
                else:
                    break
            print("Loaded scenario tree solution for %s time stages "
                  "from file %s" % (cntr, self.get_option('input_name')))
            return True

        print("No value was set for %s option 'input_name'. "
              "Nothing will be saved." % (type(self).__name__))
        return False

def extract_node_solution(tree_node):
    solution = {}
    for variable_id in tree_node._standard_variable_ids:
        varname, index = tree_node._variable_ids[variable_id]
        # store variable solution data as a list of (index, value)
        # tuples We avoid nesting another dictionary mapping index ->
        # value because (a) its cheaper and more lightweight as a list
        # and (b) because json serializes all dictionary keys as
        # strings (meaning an index of None is not recoverable)
        if varname not in solution:
            solution[varname] = []
        if variable_id in tree_node._solution:
            solution[varname].append((index, tree_node._solution[variable_id]))
        else:
            name, index = tree_node._variable_ids[variable_id]
            full_name = name+indexToString(index)
            print("%s: node solution missing for variable with scenario tree "
                  "id %s (%s)"
                  % (tree_node.name, variable_id, full_name))
            return None
    for varname in list(solution.keys()):
        solution[varname] = sorted(solution[varname], key=lambda x: x[0])
    return solution

class JSONSolutionSaverExtension(PySPConfiguredExtension,
                                 PySPConfiguredObject,
                                 SingletonPlugin):

    implements(IPySPSolutionSaverExtension)

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "output_name")
        safe_declare_common_option(options,
                                   "save_stages")

        return options

    _default_options_prefix = "jsonsaver_"

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

    def save(self, manager):

        if self.get_option("output_name") is not None:
            stage_solutions = []
            # Do NOT open file in 'binary' mode when dumping JSON
            # (produces an error in Python3)
            with open(self.get_option('output_name'), 'w') as f:
                cntr = 0
                for stage in manager.scenario_tree.stages:
                    if (self.get_option('save_stages') <= 0) or \
                       (cntr+1 <= self.get_option('save_stages')):
                        cntr += 1
                        node_solutions = {}
                        for tree_node in stage.nodes:
                            _node_solution = extract_node_solution(tree_node)
                            if _node_solution is None:
                                print("No solution appears to be stored in node with "
                                      "name %s. No solution will be saved."
                                      % (tree_node.name))
                                return False
                            node_solutions[tree_node.name] = _node_solution
                        stage_solutions.append(node_solutions)
                    else:
                        break
                json.dump(stage_solutions, f, indent=2, sort_keys=True)
            print("Saved scenario tree solution for %s time stages "
                  "to file %s" % (cntr, self.get_option('output_name')))
            return True

        print("No value was set for %s option 'output_name'. "
              "Nothing will be saved." % (type(self).__name__))
        return False
