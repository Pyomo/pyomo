#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import copy
import six
import pickle
import shelve
import json

from pyutilib.misc import ArchiveReaderFactory

from pyomo.util.plugin import *
from pyomo.pysp import phextension
from pyomo.pysp.phutils import indexToString
from pyomo.pysp.phsolverserverutils import TransmitType
import pyomo.solvers.plugins.smanager

from six import iteritems

bytes_cast = lambda x:x
if six.PY3:
    bytes_cast = lambda x: x.encode()

# TODO:
#   - snapshot node _fix_queue

def extract_convergence(ph):
    metric = ph._convergers[0].lastMetric() \
             if len(ph._convergers[0]._metric_history) \
                else None
    convergence = {'metric':metric,
                   'fixed variable counts':\
                     {'continuous':ph._total_fixed_continuous_vars,
                      'discrete':ph._total_fixed_discrete_vars},
                   'blended variable counts':\
                     {'continuous':ph._total_continuous_vars,
                      'discrete':ph._total_discrete_vars}}
    return convergence

def extract_scenario_tree_structure(scenario_tree):
    scenario_tree_structure = {}
    scenario_tree_structure['scenarios'] = {}
    for scenario in scenario_tree._scenarios:
        scenario_structure = \
            scenario_tree_structure['scenarios'][scenario._name] = {}
        scenario_structure['name'] = scenario._name
        scenario_structure['probability'] = scenario._probability
        scenario_structure['nodes'] = \
            [node._name for node in scenario._node_list]
    scenario_tree_structure['stages'] = {}
    for stage_order, stage in enumerate(scenario_tree._stages):
        stage_structure = scenario_tree_structure['stages'][stage._name] = {}
        stage_structure['name'] = stage._name
        stage_structure['nodes'] = [node._name for node in stage._tree_nodes]
        stage_structure['order'] = stage_order
    scenario_tree_structure['nodes'] = {}
    for tree_node in scenario_tree._tree_nodes:
        node_structure = \
            scenario_tree_structure['nodes'][tree_node._name] = {}
        parent = tree_node._parent
        node_structure['name'] = tree_node._name
        node_structure['parent'] = \
            parent._name if (parent is not None) else None
        node_structure['children'] = \
            [child_node._name for child_node in tree_node._children]
        node_structure['stage'] = tree_node._stage._name
        node_structure['conditional probability'] = \
            tree_node._conditional_probability
        node_structure['probability'] = tree_node._probability
        node_structure['scenarios'] = \
            [node_scenario._name for node_scenario in tree_node._scenarios]
    return scenario_tree_structure

def extract_scenario_solutions(scenario_tree,
                               include_ph_objective_parameters=False,
                               include_leaf_stage_vars=True):
    scenario_solutions = {}
    for scenario in scenario_tree._scenarios:
        scenario_name = scenario._name
        scenario_sol = scenario_solutions[scenario_name] = {}
        variable_sol = scenario_sol['variables'] = {}
        for tree_node in scenario._node_list:
            isNotLeafNode = not tree_node.is_leaf_node()
            if isNotLeafNode or include_leaf_stage_vars:
                if isNotLeafNode and include_ph_objective_parameters:
                    weight_values = scenario._w[tree_node._name]
                    rho_values = scenario._rho[tree_node._name]
                x_values = scenario._x[tree_node._name]
                for variable_id, (var_name, index) in \
                      iteritems(tree_node._variable_ids):
                    name_label = str(var_name)+str(indexToString(index))
                    varsol = variable_sol[name_label] = {}
                    varsol['value'] = x_values.get(variable_id)
                    varsol['fixed'] = scenario.is_variable_fixed(tree_node,
                                                                 variable_id)
                    varsol['stale'] = scenario.is_variable_stale(tree_node,
                                                                 variable_id)

                    if include_ph_objective_parameters:
                        if isNotLeafNode and \
                           (variable_id in tree_node._standard_variable_ids):
                            varsol['rho'] = rho_values[variable_id] \
                                            if (isNotLeafNode) \
                                               else None
                            varsol['weight'] = weight_values[variable_id] \
                                               if (isNotLeafNode) \
                                                  else None
                        else:
                            varsol['rho'] = None
                            varsol['weight'] = None

        scenario_sol['objective'] = scenario._objective
        scenario_sol['cost'] = scenario._cost

        if include_ph_objective_parameters:
            scenario_sol['ph weight term'] = scenario._weight_term_cost
            scenario_sol['ph proximal term'] = scenario._proximal_term_cost

        scenario_sol['stage costs'] = copy.deepcopy(scenario._stage_costs)

    return scenario_solutions

def extract_node_solutions(scenario_tree,
                           include_ph_objective_parameters=False,
                           include_variable_statistics=False,
                           include_leaf_stage_vars=True):

    scenario_tree.snapshotSolutionFromScenarios()
    node_solutions = {}
    stages = None
    if include_leaf_stage_vars:
        stages = scenario_tree._stages
    else:
        stages = scenario_tree._stages[:-1]
    for stage in stages:
        for tree_node in stage._tree_nodes:
            isNotLeafNode = not tree_node.is_leaf_node()
            node_sol = node_solutions[tree_node._name] = {}
            variable_sol = node_sol['variables'] = {}
            for variable_id, (var_name, index) in \
                   iteritems(tree_node._variable_ids):
                name_label = str(var_name)+str(indexToString(index))
                sol = variable_sol[name_label] = {}
                sol['solution'] = tree_node._solution[variable_id]
                sol['fixed'] = tree_node.is_variable_fixed(variable_id)
                sol['derived'] = \
                    bool(variable_id in tree_node._derived_variable_ids)
                if include_variable_statistics:
                    if isNotLeafNode:
                        sol['minimum'] = tree_node._minimums[variable_id]
                        sol['average'] = tree_node._averages[variable_id]
                        sol['maximum'] = tree_node._maximums[variable_id]
                    else:
                        sol['minimum'] = None
                        sol['average'] = None
                        sol['maximum'] = None
                if include_ph_objective_parameters:
                    if isNotLeafNode and \
                       (variable_id in tree_node._standard_variable_ids):
                        sol['xbar'] = tree_node._xbars[variable_id]
                        sol['wbar'] = tree_node._wbars[variable_id]
                    else:
                        sol['xbar'] = None
                        sol['wbar'] = None
            node_sol['expected cost'] = tree_node.computeExpectedNodeCost()
    return node_solutions

#
# A PH warmstart consists of values for W and XBAR
# (nothing more)
#
def load_ph_warmstart(ph,
                      scenariotree_solution):

    scenario_tree = ph._scenario_tree
    scenario_solutions = scenariotree_solution['scenario solutions']
    for scenario in scenario_tree._scenarios:
        scenario_name = scenario._name
        scenario_sol = scenario_solutions[scenario_name]
        variable_sol = scenario_sol['variables']
        for tree_node in scenario._node_list:
            isNotLeafNode = not tree_node.is_leaf_node()
            if isNotLeafNode:
                scenario._w[tree_node.name].clear()
                scenario_w = scenario._w[tree_node._name]
                for variable_id, (var_name, index) in \
                    iteritems(tree_node._variable_ids):
                    name_label = str(var_name)+str(indexToString(index))
                    varsol = variable_sol[name_label]
                    if variable_id in tree_node._standard_variable_ids:
                        if 'weight' in varsol:
                            scenario_w[variable_id] = varsol['weight']

    node_solutions = scenariotree_solution['node solutions']
    for stage in scenario_tree._stages[:-1]:
        for tree_node in stage._tree_nodes:
            variable_sol = node_solutions[tree_node._name]['variables']
            for variable_id in tree_node._standard_variable_ids:
                var_name, index = tree_node._variable_ids[variable_id]
                sol = variable_sol[str(var_name)+str(indexToString(index))]
                tree_node._xbars[variable_id] = sol['xbar']

def _dump_to_history(filename,
                     data,
                     key,
                     last=False,
                     first=False,
                     use_json=False):

    assert not (first and last)
    if use_json:
        file_string = 'wb' if first else \
                      'ab+'
        with open(filename, file_string) as f:
            if first:
                f.write(bytes_cast('{\n'))
            else:
                # make sure we are at the end of the file
                f.seek(0,2)
                # overwrite the previous \n}\n
                f.truncate(f.tell()-3)
                f.write(bytes_cast(',\n'))
            f.write(bytes_cast('  "'+key+'":\n'))
            f.write(bytes_cast(json.dumps(data,indent=2)))
            f.write(bytes_cast('\n}\n'))
    else:
        if first:
            flag = 'n'
        else:
            flag = 'c'
        d = shelve.open(filename,
                        flag=flag,
                        protocol=pickle.HIGHEST_PROTOCOL)
        d[key] = data
        if first:
            d['results keys'] = []
        if key != 'scenario tree':
            d['results keys'] += [key]
        d.close()

class phhistoryextension(SingletonPlugin):

    implements(phextension.IPHExtension)

    # the below is a hack to get this extension into the
    # set of IPHExtension objects, so it can be queried
    # automagically by PH.
    alias("PHHistoryExtension")

    def __init__(self):
        self._history_started = False
        self._ph_history_filename = "ph_history"
        self._use_json = int(os.environ.get("PHHISTORYEXTENSION_USE_JSON",0))
        if self._use_json:
            self._ph_history_filename += ".json"
        else:
            self._ph_history_filename += ".db"
        self._history_offset = 0

    def reset(self, ph):
        self.__init__()

    def pre_ph_initialization(self,ph):
        pass

    def post_instance_creation(self,ph):
        pass

    def post_ph_initialization(self, ph):

        # TODO: Add a print statement notifying the user of this change
        # Make sure we transmit at least all the ph variables on the
        # scenario tree (including leaf nodes). If the default
        # has already been set to transmit more, then we are fine.
        # (hence the |=)
        if isinstance(ph._solver_manager,
                      pyomo.solvers.plugins.\
                      smanager.phpyro.SolverManager_PHPyro):
            print("Overriding default variable transmission settings "
                  "for PHPyro to transmit leaf-stage variable values "
                  "at intermediate PH iterations.")
            ph._phpyro_variable_transmission_flags |= \
                TransmitType.all_stages
            ph._phpyro_variable_transmission_flags |= \
                TransmitType.blended
            ph._phpyro_variable_transmission_flags |= \
                TransmitType.derived
            ph._phpyro_variable_transmission_flags |= \
                TransmitType.fixed
            ph._phpyro_variable_transmission_flags |= \
                TransmitType.stale

    def post_iteration_0_solves(self, ph):
        pass

    def post_iteration_0(self, ph):
        pass

    def _prepare_history_file(self, ph):

        if not self._history_started:
            if (ph._ph_warmstarted) and \
               (ph._ph_warmstart_file is not None):
                assert ph._ph_warmstart_index is not None
                self._ph_history_file = ph._ph_warmstart_file
                self._history_offset = int(ph._ph_warmstart_index) + 1
                print("Detected PH warmstart file. Appending to "
                      "content and storing new iterations with offset. "
                      "First new iteration will be saved with index: %s\n"
                      % (self._history_offset))
                self._history_started = True
            else:
                data = extract_scenario_tree_structure(ph._scenario_tree)
                _dump_to_history(self._ph_history_filename,
                                 data,
                                 'scenario tree',
                                 first=True,
                                 use_json=self._use_json)
                self._history_started = True

    def _snapshot_all(self, ph):
        data = {}
        data['convergence'] = extract_convergence(ph)
        data['scenario solutions'] = \
            extract_scenario_solutions(ph._scenario_tree, True)
        data['node solutions'] = \
            extract_node_solutions(ph._scenario_tree, True, True)
        return data

    def pre_iteration_k_solves(self, ph):
        self._prepare_history_file(ph)
        key = str(ph._current_iteration - 1 + self._history_offset)
        data = self._snapshot_all(ph)
        _dump_to_history(self._ph_history_filename,
                         data,
                         key,
                         use_json=self._use_json)

    def post_iteration_k_solves(self, ph):
        pass

    def post_iteration_k(self, ph):
        pass

    def post_ph_execution(self, ph):
        self._prepare_history_file(ph)
        key = str(ph._current_iteration + \
                  self._history_offset)
        data = self._snapshot_all(ph)
        _dump_to_history(self._ph_history_filename,
                         data,
                         key,
                         last=True,
                         use_json=self._use_json)
        print("PH algorithm history written to file="
              +self._ph_history_filename)

def load_history(filename):

    with ArchiveReaderFactory(filename) as archive:

        outf = archive.extract()

        history = None
        try:
            with open(outf) as f:
                history = json.load(f)
        except:
            history = None
            try:
                history = shelve.open(outf,
                                      flag='r')
            except:
                history = None

        if history is None:
            raise RuntimeError("Unable to open ph history file as JSON "
                               "or python Shelve DB format")

        scenario_tree_dict = history['scenario tree']

        try:
            iter_keys = history['results keys']
        except KeyError:
            # we are using json format (which loads the entire file
            # anyway)
            iter_keys = list(history.keys())
            iter_keys.remove('scenario tree')

        iterations = sorted(int(k) for k in iter_keys)
        iterations = [str(k) for k in iterations]

    return scenario_tree_dict, history, iterations

def load_solution(filename):

    with ArchiveReaderFactory(filename) as archive:

        outf = archive.extract()

        solution = None
        try:
            with open(outf) as f:
                solution = json.load(f)
        except:
            solution = None
            try:
                solution = shelve.open(outf,
                                      flag='r')
            except:
                solution = None

        if solution is None:
            raise RuntimeError("Unable to open ph solution file as JSON "
                               "or python Shelve DB format")

        scenario_tree_dict = solution['scenario tree']
        solution.pop('scenario tree')

    return scenario_tree_dict, solution
