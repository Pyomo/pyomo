#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('ScenarioTreeNode',
           'ScenarioTreeStage',
           'Scenario',
           'ScenarioTreeBundle',
           'ScenarioTree')

import sys
import random
import copy
import math
import logging

from pyomo.common.collections import ComponentMap, OrderedDict
from pyomo.core import (value, minimize, maximize,
                        Var, Expression, Block,
                        Objective, SOSConstraint,
                        ComponentUID)
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.repn import generate_standard_repn
from pyomo.pysp.phutils import (BasicSymbolMap,
                                indexToString,
                                isVariableNameIndexed,
                                extractVariableNameAndIndex,
                                extractComponentIndices,
                                find_active_objective)

from six import iterkeys, iteritems, itervalues
from six.moves import xrange

logger = logging.getLogger('pyomo.pysp')

CUID_repr_version = 1

class _CUIDLabeler(object):
    def __init__(self):
        self._cuid_map = ComponentMap()

    def update_cache(self, block):
        self._cuid_map.update(
            ComponentUID.generate_cuid_string_map(
                block, repr_version=CUID_repr_version))

    def clear_cache(self):
        self._cuid_map = {}

    def __call__(self, obj):
        if obj in self._cuid_map:
            return self._cuid_map[obj]
        else:
            cuid = ComponentUID(obj).get_repr(version=1)
            self._cuid_map[obj] = cuid
            return cuid

class ScenarioTreeNode(object):

    """ Constructor
    """

    VARIABLE_FIXED = 0
    VARIABLE_FREED = 1

    def __init__(self, name, conditional_probability, stage):

        # self-explanatory!
        self._name = name

        # the stage to which this tree node belongs.
        self._stage = stage

        # defines the tree structure
        self._parent = None

        # a collection of ScenarioTreeNodes
        self._children = []

        # conditional on parent
        self._conditional_probability = conditional_probability

        # a collection of all Scenario objects passing through this
        # node in the tree
        self._scenarios = []

        # the cumulative probability of scenarios at this node.
        # cached for efficiency.
        self._probability = 0.0

        # a map between a variable name and a list of original index
        # match templates, specified as strings.  we want to maintain
        # these for a variety of reasons, perhaps the most important
        # being that for output purposes. specific indices that match
        # belong to the tree node, as that may be specific to a tree
        # node.
        self._variable_templates = {}
        self._derived_variable_templates = {}

        #
        # information relating to all variables blended at this node, whether
        # of the standard or derived varieties.
        #
        # maps id -> (name, index)
        self._variable_ids = {}
        # maps (name,index) -> id
        self._name_index_to_id = {}
        # maps id -> list of (vardata,probability) across all scenarios
        self._variable_datas = {}

        # keep track of the variable indices at this node, independent
        # of type.  this is useful for iterating. maps variable name
        # to a list of indices.
        self._variable_indices = {}

        # variables are either standard or derived - but not both.
        # partition the ids into two sets, as we deal with these
        # differently in algorithmic and reporting contexts.
        self._standard_variable_ids = set()
        self._derived_variable_ids = set()
        # A temporary solution to help wwphextension and other code
        # for when pyomo instances no longer live on the master node
        # when using PHPyro
        self._integer = set()
        self._binary = set()
        self._semicontinuous = set()

        # a tuple consisting of (1) the name of the variable that
        # stores the stage-specific cost in all scenarios and (2) the
        # corresponding index *string* - this is converted in the tree
        # node to a real index.
        # TODO: Change the code so that this is a ComponentUID string
        self._cost_variable = None

        # a list of _VarData objects, representing the cost variables
        # for each scenario passing through this tree node.
        # NOTE: This list actually contains tuples of
        #       (_VarData, scenario-probability) pairs.
        self._cost_variable_datas = []

        # general use statistics for the variables at each node.
        # each attribute is a map between the variable name and a
        # parameter (over the same index set) encoding the corresponding
        # statistic computed over all scenarios for that node. the
        # parameters are named as the source variable name suffixed
        # by one of: "NODEMIN", "NODEAVG", and "NODEMAX".
        # NOTE: the averages are probability_weighted - the min/max
        #       values are not.
        # NOTE: the parameter names are basically irrelevant, and the
        #       convention is assumed to be enforced by whoever populates
        #       these parameters.
        self._minimums = {}
        self._averages = {}
        self._maximums = {}
        # This gets pushed into PHXBAR on the instances
        self._xbars = {}
        # This gets pushed into PHBLEND on the instances
        self._blend = {}
        self._wbars = {} # USED IN THE DUAL
        # node variables ids that are fixed (along with the value to fix)
        self._fixed = {}
        # variable ids currently out of sync with instance data
        # variable_id -> VARIABLE_FIXED | VARIABLE_FREED
        self._fix_queue = {}

        # solution (variable) values for this node. assumed to be distinct
        # from self._averages, as the latter are not necessarily feasible.
        # keys are variable ids.
        self._solution = {}

    @property
    def name(self):
        return self._name

    @property
    def stage(self):
        return self._stage

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return tuple(self._children)

    @property
    def scenarios(self):
        return self._scenarios

    @property
    def conditional_probability(self):
        return self._conditional_probability

    @property
    def probability(self):
        return self._probability

    #
    # Updates the minimum, maximum, and average for this node
    # from the solutions stored on the scenario objects
    #
    def updateNodeStatistics(self):

        scenario_solutions = \
            [(scenario._probability, scenario._x[self._name]) \
             for scenario in self._scenarios]

        for variable_id in self._variable_ids:

            stale = False
            values = []
            avg_value = 0.0
            for probability, var_values in scenario_solutions:
                val = var_values[variable_id]
                if val is not None:
                    avg_value += probability * val
                    values.append(val)
                else:
                    stale = True
                    break

            if stale:
                self._minimums[variable_id] = None
                self._maximums[variable_id] = None
                self._averages[variable_id] = None
            else:
                avg_value /= self._probability
                self._minimums[variable_id] = min(values)
                self._maximums[variable_id] = max(values)
                self._averages[variable_id] = avg_value

    #
    # given a set of scenario instances, compute the set of indices
    # for non-anticipative variables at this node, as defined by the
    # input match templates.
    #

    def updateVariableIndicesAndValues(self,
                                       component_name,
                                       match_templates,
                                       derived=False,
                                       id_labeler=None,
                                       name_index_to_id_map=None):

        # ensure that the variable exists on each scenario instance,
        # and that there is at least one index match per template.

        # To avoid calling extractComponentIndices more than necessary
        # we take the last scenario in the next loop as our
        # "representative" scenario from which we use the
        # new_match_indices list
        new_match_indices = None
        var_component = {}
        symbolmap = {}
        scenario = None
        isVar = False
        for scenario in self._scenarios:

            scenario_instance = scenario._instance

            if scenario_instance is None:
                continue

            component_object = \
                scenario_instance.find_component(component_name)
            if component_object is None:
                raise RuntimeError(
                    "The component=%s associated with stage=%s "
                    "is not present in instance=%s"
                    % (component_name,
                       self._stage._name,
                       scenario_instance.name))

            if component_object.ctype is not Block:
                isVar = (component_object.ctype is Var)
                if not derived:
                    if not isVar:
                        raise RuntimeError("The component=%s "
                                           "associated with stage=%s "
                                           "is present in instance=%s "
                                           "but is not a variable - type=%s"
                                           % (component_name,
                                              self._stage._name,
                                              scenario_instance.name,
                                              type(component_object)))
                else:
                    if (not isVar) and \
                       (component_object.ctype is not Expression) and \
                       (component_object.ctype is not Objective):
                        raise RuntimeError("The derived component=%s "
                                           "associated with stage=%s "
                                           "is present in instance=%s "
                                           "but is not a Var or Expression "
                                           "- type=%s"
                                           % (component_name,
                                              self._stage._name,
                                              scenario_instance.name,
                                              type(component_object)))
            else:
                tmp_match_template = ("",)
                for match_template in match_templates:
                    for index in extractComponentIndices(component_object,
                                                         match_template):
                        # extract all variables from this block
                        if component_object[index].active:
                            for variable in component_object[index].\
                                   component_objects(Var,
                                                     descend_into=True):
                                self.updateVariableIndicesAndValues(
                                    variable.name,
                                    tmp_match_template,
                                    derived=derived,
                                    id_labeler=id_labeler,
                                    name_index_to_id_map=name_index_to_id_map)
                return

            new_match_indices = []

            for match_template in match_templates:

                indices = extractComponentIndices(component_object,
                                                 match_template)

                # validate that at least one of the indices in the
                # variable matches to the template - otherwise, the
                # template is bogus.  with one exception: if the
                # variable is empty (the index set is empty), then
                # don't warn - the instance was designed this way.
                if (len(indices) == 0) and (len(component_object) > 0):
                    raise ValueError("No indices match template=%s "
                                     "for variable=%s in scenario=%s"
                                     % (match_template,
                                        component_name,
                                        scenario.name))

                new_match_indices.extend(indices)

            var_component[scenario._name] = \
                scenario_instance.find_component(component_name)

            if (id_labeler is not None) or \
               (name_index_to_id_map is not None):
                # Tag each instance with a ScenarioTreeSymbolMap. This
                # will allow us to identify common blended variables
                # within a node across scenario instances without
                # having to do an expensive name lookup each time.
                this_symbolmap = getattr(scenario_instance,
                                         "_ScenarioTreeSymbolMap",
                                         None)
                if this_symbolmap is None:
                    this_symbolmap = \
                        scenario_instance._ScenarioTreeSymbolMap = \
                            BasicSymbolMap()
                symbolmap[scenario._name] = this_symbolmap

        # find a representative scenario instance belonging to (or
        # passing through) this node in the tree. the first scenario
        # is as good as any.
        # NOTE: At some point we should check that the index sets
        #       across all scenarios at a node actually match for each
        #       variable.
        self._variable_indices.setdefault(
            component_name, []).extend(new_match_indices)

        # cache some stuff up-front - we're accessing these
        # attributes a lot in the loops below.
        if not derived:
            variable_ids_to_update = self._standard_variable_ids
        else:
            variable_ids_to_update = self._derived_variable_ids

        self_variable_ids = self._variable_ids
        self_variable_datas = self._variable_datas

        if (id_labeler is not None) or \
           (name_index_to_id_map is not None):

            for index in sorted(new_match_indices):

                # create the ScenarioTree integer id for this variable
                # across all scenario instances, or look it up if a
                # map has been provided.
                reference_object = \
                    var_component[self._scenarios[0].name][index]
                scenario_tree_id = None
                if id_labeler != None:
                    scenario_tree_id = id_labeler(reference_object)
                elif name_index_to_id_map != None:
                    scenario_tree_id = \
                        name_index_to_id_map[component_name, index]

                variable_ids_to_update.add(scenario_tree_id)

                self_variable_ids[scenario_tree_id] = (component_name,index)
                self._name_index_to_id[(component_name,index)] = \
                    scenario_tree_id
                self_variable_datas[scenario_tree_id] = []
                for scenario in self._scenarios:
                    vardata = var_component[scenario._name][index]
                    symbolmap[scenario._name].addSymbol(vardata,
                                                        scenario_tree_id)
                    self_variable_datas[scenario_tree_id].append(
                        (vardata, scenario._probability))
                # We are trusting that each instance variable has the same
                # domain (as we always do)
                if isVar:
                    vardata = self_variable_datas[scenario_tree_id][0][0]
                    if vardata.is_integer():
                        self._integer.add(scenario_tree_id)
                    if vardata.is_binary():
                        self._binary.add(scenario_tree_id)
                    # TODO
                    #if vardata.is_semicontinuous():
                    #    self._semicontinuous.add(scenario_tree_id)

    #
    # same as the above, but specialized to cost variables.
    #

    def updateCostVariableIndexAndValue(self,
                                        cost_variable_name,
                                        cost_variable_index):

        # ensure that the cost variable exists on each scenario
        # instance, and that the index is valid.  if so, add it to the
        # list of _VarDatas for scenarios at this tree node.
        for scenario in self._scenarios:
            scenario_instance = scenario._instance
            cost_variable = \
                scenario_instance.find_component(cost_variable_name)

            if cost_variable is None:
                raise ValueError("Cost variable=%s associated with "
                                 "stage=%s is not present in model=%s; "
                                 "scenario tree construction failed"
                                 % (cost_variable_name,
                                    self._stage._name,
                                    scenario_instance.name))
            if not cost_variable.ctype in [Var,Expression,Objective]:
                raise RuntimeError("The component=%s associated with stage=%s "
                                   "is present in model=%s but is not a "
                                   "variable or expression - type=%s"
                                   % (cost_variable_name,
                                      self._stage._name,
                                      scenario_instance.name,
                                      cost_variable.ctype))
            if cost_variable_index not in cost_variable:
                raise RuntimeError("The index %s is not defined for cost "
                                   "variable=%s on model=%s"
                                   % (cost_variable_index,
                                      cost_variable_name,
                                      scenario_instance.name))
            self._cost_variable_datas.append(
                (cost_variable[cost_variable_index],
                 scenario._probability))

    #
    # given a set of scenario instances, compute the set of indices
    # being blended for each variable at this node. populates the
    # _variable_indices and _variable_values attributes of a tree
    # node.
    #

    def populateVariableIndicesAndValues(self,
                                         id_labeler=None,
                                         name_index_to_id_map=None,
                                         initialize_solution_data=True):
        self._variable_indices = {}
        self._variable_datas = {}
        self._standard_variable_ids = set()
        self._derived_variable_ids = set()

        stage_variables = self._stage._variable_templates
        for component_name in sorted(iterkeys(stage_variables)):
            self.updateVariableIndicesAndValues(
                component_name,
                stage_variables[component_name],
                derived=False,
                id_labeler=id_labeler,
                name_index_to_id_map=name_index_to_id_map)
        node_variables = self._variable_templates
        for component_name in sorted(iterkeys(node_variables)):
            self.updateVariableIndicesAndValues(
                component_name,
                node_variables[component_name],
                derived=False,
                id_labeler=id_labeler,
                name_index_to_id_map=name_index_to_id_map)

        stage_derived_variables = self._stage._derived_variable_templates
        for component_name in sorted(iterkeys(stage_derived_variables)):
            self.updateVariableIndicesAndValues(
                component_name,
                stage_derived_variables[component_name],
                derived=True,
                id_labeler=id_labeler,
                name_index_to_id_map=name_index_to_id_map)
        node_derived_variables = self._derived_variable_templates
        for component_name in sorted(iterkeys(node_derived_variables)):
            self.updateVariableIndicesAndValues(
                component_name,
                node_derived_variables[component_name],
                derived=True,
                id_labeler=id_labeler,
                name_index_to_id_map=name_index_to_id_map)

        self.updateCostVariableIndexAndValue(self._cost_variable[0],
                                             self._cost_variable[1])

        if not initialize_solution_data:
            return

        # Create a fully populated scenario tree node.
        if not self.is_leaf_node():
            self._minimums = dict.fromkeys(self._variable_ids,0)
            self._maximums = dict.fromkeys(self._variable_ids,0)
            # this is the true variable average at the node (unmodified)
            self._averages = dict.fromkeys(self._variable_ids,0)
            # this is the xbar used in the PH objective.
            self._xbars = dict.fromkeys(self._standard_variable_ids,None)
            # this is the blend used in the PH objective
            self._blend = dict.fromkeys(self._standard_variable_ids,None)
            # For the dual ph algorithm
            self._wbars = dict.fromkeys(self._standard_variable_ids,None)

            for scenario in self._scenarios:

                scenario._w[self._name] = \
                    dict.fromkeys(self._standard_variable_ids,None)
                scenario._rho[self._name] = \
                    dict.fromkeys(self._standard_variable_ids,None)

        for scenario in self._scenarios:
            scenario._x[self._name] = \
                dict.fromkeys(self._variable_ids,None)

    #
    # copies the parameter values values from the _averages attribute
    # into the _solution attribute - only for active variable values.
    # for leaf nodes, simply copies the values from the _VarValue objects
    # at that node - because there are no statistics maintained.
    #

    def snapshotSolutionFromAverages(self):
        self._solution = {}
        if self.is_leaf_node():
            self._solution.update(self._scenarios[0]._x[self._name])
        else:
            self._solution.update(self._averages)

    #
    # computes the solution values from the composite scenario
    # solutions at this tree node.
    #

    # Note: Trying to work this function out of the code. The only solution
    #       we should get used to working with is that stored on the scenario
    #       objects
    def XsnapshotSolutionFromInstances(self):

        self._solution = {}

        for variable_id in self._standard_variable_ids:

            var_datas = self._variable_datas[variable_id]
            # the following loop is just a sanity check.
            for var_data, scenario_probability in var_datas:
                # a variable that is fixed will be flagged as unused.
                if (var_data.stale) and (not var_data.fixed):
                    # Note: At this point the only way to get the name
                    #       of the scenario for this specific vardata
                    #       in general is to print its full name
                    # This will either be "MASTER", the bundle name,
                    # or the scenario name The important thing is that
                    # we always have the scenario name somewhere in
                    # the variable name we print
                    model_name = var_data.model().name
                    full_name = model_name+"."+var_data.name
                    if not self.is_leaf_node():
                        print("CAUTION: Encountered variable=%s "
                              "on node %s that is not in use within its "
                              "respective scenario instance but the scenario tree "
                              "specification indicates that non-anticipativity is to "
                              "be enforced; the variable should either be eliminated "
                              "from the model or from the scenario tree specification."
                              % (full_name, self._name))
                    else:
                        print("CAUTION: Encountered variable=%s "
                              "on leaf node %s that is not in use within "
                              "its respective scenario instance. This can be indicative "
                              "of a modeling error; the variable should either be "
                              "eliminated from the model or from the scenario tree "
                              "specification." % (full_name, self._name))

            # if a variable is stale, it could be because it is fixed,
            # in which case, we want to snapshot the average value
            avg = sum(scenario_probability * value(var_data)
                      for var_data, scenario_probability in var_datas
                      if (not var_data.stale) or var_data.fixed)

            # the node probability is allowed to be zero in the
            # scenario tree specification.  this is useful in cases
            # where one wants to temporarily ignore certain scenarios.
            # in this case, just skip reporting of variables for that
            # node.
            if self._probability > 0.0:
                avg /= self._probability

            self._solution[variable_id] = avg

        for variable_id in self._derived_variable_ids:

            var_datas = self._variable_datas[variable_id]

            avg = sum(scenario_probability * value(var_data)
                      for var_data, scenario_probability in var_datas)

            # the node probability is allowed to be zero in the
            # scenario tree specification.  This is useful in cases
            # where one wants to temporarily ignore certain scenarios.
            # in this case, just skip reporting of variables for that
            # node.
            if self._probability > 0.0:
                avg /= self._probability

            self._solution[variable_id] = avg

    def snapshotSolutionFromScenarios(self):

        self._solution = {}

        for variable_id in self._variable_ids:

            var_values = []
            avg = 0.0
            for scenario in self._scenarios:
                val = scenario._x[self.name].get(variable_id)
                if val is not None:
                    var_values.append((val, scenario._probability))
                else:
                    avg = None
                    break

            if avg is not None:

                # the following loop is just a sanity check.
                for scenario in self._scenarios:
                    scenario_probability = scenario._probability
                    var_value = scenario._x[self._name][variable_id]
                    is_fixed = scenario.is_variable_fixed(self, variable_id)
                    is_stale = scenario.is_variable_stale(self, variable_id)
                    # a variable that is fixed will be flagged as unused.
                    if is_stale and (not is_fixed):
                        variable_name, index = self._variable_ids[variable_id]
                        full_name = variable_name+indexToString(index)
                        if not self.is_leaf_node():
                            print("CAUTION: Encountered variable=%s "
                                  "on node %s that is not in use within its "
                                  "respective scenario %s but the scenario tree "
                                  "specification indicates that non-anticipativity is "
                                  "to be enforced; the variable should either be "
                                  "eliminated from the model or from the scenario "
                                  "tree specification." % (full_name,
                                                           self._name,
                                                           scenario._name))
                        else:
                            print("CAUTION: Encountered variable=%s "
                                  "on leaf node %s that is not in use within "
                                  "its respective scenario %s. This can be indicative "
                                  "of a modeling error; the variable should either be "
                                  "eliminated from the model or from the scenario "
                                  "tree specification." % (full_name,
                                                           self._name,
                                                           scenario._name))
                    else:
                        avg += scenario_probability * var_value

                # the node probability is allowed to be zero in the
                # scenario tree specification.  this is useful in cases
                # where one wants to temporarily ignore certain scenarios.
                # in this case, just skip reporting of variables for that
                # node.
                if self._probability > 0.0:
                    avg /= self._probability

            self._solution[variable_id] = avg

    #
    # a utility to compute the cost of the current node plus the expected costs of child nodes.
    #

    def computeExpectedNodeCost(self):

        stage_name = self._stage._name
        if any(scenario._stage_costs[stage_name] is None \
               for scenario in self._scenarios):
            return None

        my_cost = self._scenarios[0]._stage_costs[stage_name]
        # Don't assume the node has converged, this can
        # result in misleading output
        # UPDATE: It turns out this entire function is misleading
        #         it will be removed
        """
        my_cost = sum(scenario._stage_costs[stage_name] * scenario._probability \
                      for scenario in self._scenarios)
        my_cost /= sum(scenario._probability for scenario in self._scenarios)
        """
        # This version implicitly assumes convergence (which can be garbage for ph)

        children_cost = 0.0
        for child in self._children:
            child_cost = child.computeExpectedNodeCost()
            if child_cost is None:
                return None
            else:
                children_cost += (child._conditional_probability * child_cost)
        return my_cost + children_cost

    #
    # a simple predicate to check if this tree node belongs to the
    # last stage in the scenario tree.
    #
    def is_leaf_node(self):

        return self._stage.is_last_stage()

    #
    # a utility to determine if the input variable name/index pair is
    # a derived variable.
    #
    def is_derived_variable(self, variable_name, variable_index):
        return (variable_name, variable_index) in self._name_index_to_id

    #
    # a utility to extract the value for the input name/index pair.
    #
    def get_variable_value(self, name, index):

        try:
            variable_id = self._name_index_to_id[(name,index)]
        except KeyError:
            raise ValueError("No ID for variable=%s, index=%s "
                             "is defined for scenario tree "
                             "node=%s" % (name, index, self._name))

        try:
            return self._solution[variable_id]
        except KeyError:
            raise ValueError("No value for variable=%s, index=%s "
                             "is defined for scenario tree "
                             "node=%s" % (name, index, self._name))

    #
    # fix the indicated input variable / index pair to the input value.
    #
    def fix_variable(self, variable_id, fix_value):
        self._fix_queue[variable_id] = (self.VARIABLE_FIXED, fix_value)

    #
    # free the indicated input variable / index pair to the input value.
    #
    def free_variable(self, variable_id):
        self._fix_queue[variable_id] = (self.VARIABLE_FREED, None)

    def is_variable_integer(self, variable_id):
        return variable_id in self._integer

    def is_variable_binary(self, variable_id):
        return variable_id in self._binary
    is_variable_boolean = is_variable_binary

    def is_variable_semicontinuous(self, variable_id):
        return variable_id in self._semicontinuous

    def is_variable_discrete(self, variable_id):
        return self.is_variable_integer(variable_id) or \
            self.is_variable_binary(variable_id) or \
            self.is_variable_semicontinuous(variable_id)

    def is_variable_fixed(self, variable_id):
        return variable_id in self._fixed

    def push_xbar_to_instances(self):
        arbitrary_instance = self._scenarios[0]._instance
        assert arbitrary_instance != None

        # Note: the PHXBAR Param is shared amongst the
        # scenarios in a tree node, so it's only
        # necessary to grab the Param from an arbitrary
        # scenario for each node and update once
        xbar_parameter_name = "PHXBAR_"+str(self._name)
        xbar_parameter = arbitrary_instance.find_component(xbar_parameter_name)
        xbar_parameter.store_values(self._xbars)

    def push_fix_queue_to_instances(self):
        have_instances = (self._scenarios[0]._instance != None)
        for variable_id, (fixed_status, new_value) in iteritems(self._fix_queue):
            if fixed_status == self.VARIABLE_FREED:
                assert new_value is None
                if have_instances:
                    for var_data, scenario_probability in \
                        self._variable_datas[variable_id]:
                        var_data.free()
                del self._fixed[variable_id]
            elif fixed_status == self.VARIABLE_FIXED:
                if have_instances:
                    for var_data, scenario_probability in \
                        self._variable_datas[variable_id]:
                        var_data.fix(new_value)
                self._fixed[variable_id] = new_value
            else:
                raise ValueError("Unexpected fixed status %s for variable with "
                                 "scenario tree id %s" % (fixed_status,
                                                          variable_id))
        self.clear_fix_queue()

    def push_all_fixed_to_instances(self):
        have_instances = (self._scenarios[0]._instance != None)

        for variable_id, fix_value in iteritems(self._fixed):
            if have_instances:
                for var_data, scenario_probability in \
                    self._variable_datas[variable_id]:
                    var_data.fix(fix_value)
            self._fixed[variable_id] = fix_value

        self.push_fix_queue_to_instances()

    def has_fixed_in_queue(self):
        return any((v[0] == self.VARIABLE_FIXED) \
                   for v in itervalues(self._fix_queue))

    def has_freed_in_queue(self):
        return any((v[0] == self.VARIABLE_FREED) \
                   for v in itervalues(self._fix_queue))

    def clear_fix_queue(self):

        self._fix_queue.clear()

class ScenarioTreeStage(object):

    """ Constructor
    """
    def __init__(self):

        self._name = ""

        # a collection of ScenarioTreeNode objects associated with this stage.
        self._tree_nodes = []

        # the parent scenario tree for this stage.
        self._scenario_tree = None

        # a map between a variable name and a list of original index
        # match templates, specified as strings.  we want to maintain
        # these for a variety of reasons, perhaps the most important
        # being that for output purposes. specific indices that match
        # belong to the tree node, as that may be specific to a tree
        # node.
        self._variable_templates = {}

        # same as above, but for derived stage variables.
        self._derived_variable_templates = {}

        # a tuple consisting of (1) the name of the variable that
        # stores the stage-specific cost in all scenarios and (2) the
        # corresponding index *string* - this is converted in the tree
        # node to a real index.
        self._cost_variable = None

    @property
    def name(self):
        return self._name

    @property
    def nodes(self):
        return self._tree_nodes

    @property
    def scenario_tree(self):
        return self._scenario_tree

    #
    # add a new variable to the stage, which will include updating the
    # solution maps for each associated ScenarioTreeNode.
    #
    def add_variable(self,
                     variable_name,
                     new_match_template,
                     create_variable_ids=True):

        labeler = None
        if create_variable_ids is True:
            labeler = self._scenario_tree._id_labeler

        existing_match_templates = self._variable_templates.setdefault(variable_name, [])
        existing_match_templates.append(new_match_template)

        for tree_node in self._tree_nodes:
            tree_node.updateVariableIndicesAndValues(variable_name,
                                                     new_match_template,
                                                     derived=False,
                                                     id_labeler=labeler)

    #
    # a simple predicate to check if this stage is the last stage in
    # the scenario tree.
    #
    def is_last_stage(self):

        return self == self._scenario_tree._stages[-1]

class Scenario(object):

    """ Constructor
    """
    def __init__(self):

        self._name = None
        # allows for construction of node list
        self._leaf_node = None
        # sequence from parent to leaf of ScenarioTreeNodes
        self._node_list = []
        # the unconditional probability for this scenario, computed from the node list
        self._probability = 0.0
        # the Pyomo instance corresponding to this scenario.
        self._instance = None
        self._instance_cost_expression = None
        self._instance_objective = None
        self._instance_original_objective_object = None
        self._objective_sense = None
        self._objective_name = None

        # The value of the (possibly augmented) objective function
        self._objective = None
        # The value of the original objective expression
        # (which should be the sum of the stage costs)
        self._cost = None
        # The individual stage cost values
        self._stage_costs = {}
        # The value of the ph weight term piece of the objective (if it exists)
        self._weight_term_cost = None
        # The value of the ph proximal term piece of the objective (if it exists)
        self._proximal_term_cost = None
        # The value of the scenariotree variables belonging to this scenario
        # (dictionary nested by node name)
        self._x = {}
        # The value of the weight terms belonging to this scenario
        # (dictionary nested by node name)
        self._w = {}
        # The value of the rho terms belonging to this scenario
        # (dictionary nested by node name)
        self._rho = {}

        # This set of fixed or reported stale variables
        # in each tree node
        self._fixed = {}
        self._stale = {}

    @property
    def name(self):
        return self._name

    @property
    def leaf_node(self):
        return self._leaf_node

    @property
    def node_list(self):
        return tuple(self._node_list)

    @property
    def probability(self):
        return self._probability

    @property
    def instance(self):
        return self._instance

    def get_current_objective(self):
        return self._objective

    def get_current_cost(self):
        return self._cost

    def get_current_stagecost(self, stage_name):
        return self._stage_costs[stage_name]

    #
    # a utility to compute the stage index for the input tree node.
    # the returned index is 0-based.
    #

    def node_stage_index(self, tree_node):
        return self._node_list.index(tree_node)

    def is_variable_fixed(self, tree_node, variable_id):

        return variable_id in self._fixed[tree_node._name]

    def is_variable_stale(self, tree_node, variable_id):

        return variable_id in self._stale[tree_node._name]

    def update_solution_from_instance(self, stages=None):

        scenario_instance = self._instance
        scenariotree_sm_bySymbol = \
            scenario_instance._ScenarioTreeSymbolMap.bySymbol
        self._objective = self._instance_objective(exception=False)
        self._cost = self._instance_cost_expression(exception=False)
        for tree_node in self._node_list:
            cost_variable_name, cost_variable_index = \
                tree_node._cost_variable
            stage_cost_component = \
                self._instance.find_component(cost_variable_name)
            self._stage_costs[tree_node.stage.name] = \
                stage_cost_component[cost_variable_index](exception=False)
#        if abs(sum(self._stage_costs.values()) - self._cost) > 1e-6:
#            logger.warning("The value of the original objective on scenario "
#                           "%s (%s) does not equal the sum of the stage "
#                           "costs (%s) reported for that scenario."
#                           % (self.name, self._cost, sum(self._stage_costs.values())))
        self._weight_term_cost = \
            scenario_instance.PHWEIGHT_EXPRESSION(exception=False) \
            if (hasattr(scenario_instance,"PHWEIGHT_EXPRESSION") and \
                (scenario_instance.PHWEIGHT_EXPRESSION is not None)) \
            else None
        self._proximal_term_cost = \
            scenario_instance.PHPROXIMAL_EXPRESSION(exception=False) \
            if (hasattr(scenario_instance,"PHPROXIMAL_EXPRESSION") and \
                (scenario_instance.PHPROXIMAL_EXPRESSION is not None)) \
            else None

        for tree_node in self._node_list:
            if (stages is None) or (tree_node.stage.name in stages):
                # Some of these might be Expression objects so we use the
                # __call__ method rather than directly accessing .value
                # (since we want a number)
                self._x[tree_node.name].update(
                    (variable_id,
                     scenariotree_sm_bySymbol[variable_id](exception=False)) \
                    for variable_id in tree_node._variable_ids)

                scenario_fixed = self._fixed[tree_node.name]
                scenario_stale = self._stale[tree_node.name]
                scenario_fixed.clear()
                scenario_stale.clear()
                for variable_id in tree_node._variable_ids:
                    vardata = scenariotree_sm_bySymbol[variable_id]
                    if vardata.is_expression_type():
                        continue
                    if vardata.fixed:
                        scenario_fixed.add(variable_id)
                    if vardata.stale:
                        scenario_stale.add(variable_id)
            else:
                self._x[tree_node.name].clear()
                self._fixed[tree_node.name].clear()
                self._stale[tree_node.name].clear()

    def push_solution_to_instance(self):

        scenario_instance = self._instance
        scenariotree_sm_bySymbol = \
            scenario_instance._ScenarioTreeSymbolMap.bySymbol
        for tree_node in self._node_list:
            stage_name = tree_node._stage.name
            cost_variable_name, cost_variable_index = \
                tree_node._cost_variable
            stage_cost_component = \
                self._instance.find_component(cost_variable_name)[cost_variable_index]
            # Some of these might be Expression objects so we check
            # for is_expression_type before changing.value
            if not stage_cost_component.is_expression_type():
                stage_cost_component.value = self._stage_costs[stage_name]

        for tree_node in self._node_list:
            # Some of these might be Expression objects so we check
            # for is_expression_type before changing.value
            for variable_id, var_value in iteritems(self._x[tree_node._name]):
                compdata = scenariotree_sm_bySymbol[variable_id]
                if not compdata.is_expression_type():
                    compdata.value = var_value

            for variable_id in self._fixed[tree_node._name]:
                vardata = scenariotree_sm_bySymbol[variable_id]
                vardata.fix()

            for variable_id in self._stale[tree_node._name]:
                vardata = scenariotree_sm_bySymbol[variable_id]
                vardata.stale = True

    def copy_solution(self, translate_ids=None):

        solution = {}
        solution['objective'] = self._objective
        solution['cost'] = self._cost
        solution['stage costs'] = copy.deepcopy(self._stage_costs)
        solution['weight term cost'] = self._weight_term_cost
        solution['proximal term cost'] = self._proximal_term_cost
        if translate_ids is None:
            solution['x'] = copy.deepcopy(self._x)
            solution['fixed'] = copy.deepcopy(self._fixed)
            solution['stale'] = copy.deepcopy(self._stale)
        else:
            resx = solution['x'] = {}
            for tree_node_name, tree_node_x in iteritems(self._x):
                tree_node_translate_ids = translate_ids[tree_node_name]
                resx[tree_node_name] = \
                    dict((tree_node_translate_ids[scenario_tree_id],val) \
                         for scenario_tree_id, val in \
                         iteritems(tree_node_x))
            resfixed = solution['fixed'] = {}
            # NOTE: This function is frequently called to generate
            #       a set of results that is transmitted over the wire
            #       with Pyro. Some of the serializers used by Pyro4
            #       have issues with set() objects, so we convert them
            #       to tuples here and then convert them back to
            #       set objects in the set_solution() method.
            for tree_node_name, tree_node_fixed in iteritems(self._fixed):
                tree_node_translate_ids = translate_ids[tree_node_name]
                resfixed[tree_node_name] = \
                    tuple(set(tree_node_translate_ids[scenario_tree_id] \
                        for scenario_tree_id in tree_node_fixed))
            resstale = solution['stale'] = {}
            for tree_node_name, tree_node_stale in iteritems(self._stale):
                tree_node_translate_ids = translate_ids[tree_node_name]
                resstale[tree_node_name] = \
                    tuple(set(tree_node_translate_ids[scenario_tree_id] \
                        for scenario_tree_id in tree_node_stale))
        return solution

    def set_solution(self, solution):

        self._objective = solution['objective']
        self._cost = solution['cost']
        assert set(solution['stage costs'].keys()) == set(self._stage_costs.keys())
        self._stage_costs = copy.deepcopy(solution['stage costs'])
#        if abs(sum(self._stage_costs.values()) - self._cost) > 1e-6:
#            logger.warning("The value of the original objective on scenario "
#                           "%s (%s) does not equal the sum of the stage "
#                           "costs (%s) reported for that scenario."
#                           % (self.name, self._cost, sum(self._stage_costs.values())))
        self._weight_term_cost = solution['weight term cost']
        self._proximal_term_cost = solution['proximal term cost']
        assert set(solution['x'].keys()) == set(self._x.keys())
        self._x = copy.deepcopy(solution['x'])
        assert set(solution['fixed'].keys()) == set(self._fixed.keys())
        assert set(solution['stale'].keys()) == set(self._stale.keys())
        # See note in copy_solution method about converting
        # these items back to set() objects
        for node_name, fixed_ids in solution['fixed'].items():
            self._fixed[node_name] = set(fixed_ids)
        for node_name, stale_ids in solution['stale'].items():
            self._stale[node_name] = set(stale_ids)

    def push_w_to_instance(self):
        assert self._instance != None
        for tree_node in self._node_list[:-1]:
            weight_parameter_name = "PHWEIGHT_"+str(tree_node._name)
            weight_parameter = self._instance.find_component(weight_parameter_name)
            weight_parameter.store_values(self._w[tree_node._name])

    def push_rho_to_instance(self):
        assert self._instance != None

        for tree_node in self._node_list[:-1]:
            rho_parameter_name = "PHRHO_"+str(tree_node._name)
            rho_parameter = self._instance.find_component(rho_parameter_name)
            rho_parameter.store_values(self._rho[tree_node._name])

    #
    # a utility to determine the stage to which the input variable belongs.
    #

    def variableNode(self, vardata, instance=None):

        if instance is None:
            instance = vardata.parent_component().model()
        assert instance is not None

        try:
            variable_id = instance._ScenarioTreeSymbolMap.byObject[id(vardata)]
            for this_node in self._node_list:
                if variable_id in this_node._variable_ids:
                    return this_node
        except KeyError:
            for this_node in self._node_list:
                cost_variable = this_node._cost_variable
                if cost_variable[0] is not None:
                    if vardata is \
                       instance.find_component(
                           cost_variable[0])[cost_variable[1]]:
                        return this_node

        raise KeyError("Variable="+str(vardata.name)+" does "
                       "not belong to any node in the scenario tree")

    def variableNode_byNameIndex(self, variable_name, index):

        tuple_to_check = (variable_name,index)

        for this_node in self._node_list:
            if tuple_to_check in this_node._name_index_to_id:
                return this_node

        for this_node in self._node_list:
            if tuple_to_check == this_node._cost_variable:
                return this_node

        raise KeyError("Variable="+str(variable_name)+", "
                       "index="+indexToString(index)+" does not "
                       "belong to any node in the scenario tree")

    #
    # a utility to determine the stage to which the input constraint "belongs".
    # a constraint belongs to the latest stage in which referenced variables
    # in the constraint appears in that stage.
    # input is a constraint is of type "Constraint", and an index of that
    # constraint - which might be None in the case of non-indexed constraints.
    # currently doesn't deal with SOS constraints, for no real good reason.
    # returns an instance of a ScenarioTreeStage object.
    # IMPT: this method works on the standard representation ("repn" attribute)
    #       of a constraint. this implies that pre-processing of the instance
    #       has been performed.
    # NOTE: there is still the issue of whether the contained variables really
    #       belong to the same model, but that is a different issue we won't
    #       address right now (e.g., what does it mean for a constraint in an
    #       extensive form binding instance to belong to a stage?).
    #

    def constraintNode(self,
                       constraintdata,
                       repn=None,
                       instance=None,
                       assume_last_stage_if_missing=False):

        deepest_node_index = -1
        deepest_node = None

        vardata_list = None
        if isinstance(constraintdata, (SOSConstraint, _SOSConstraintData)):
            vardata_list = constraintdata.get_variables()

        else:
            if repn is None:
                repn = generate_standard_repn(constraintdata.body, quadratic=False)

            vardata_list = repn.linear_vars
            if len(repn.nonlinear_vars):
                vardata_list += repn.nonlinear_vars

        for var_data in vardata_list:

            try:
                var_node = self.variableNode(var_data, instance=instance)
            except KeyError:
                if assume_last_stage_if_missing:
                    return self._leaf_node
                model_name = var_data.model().name
                full_name = model_name+"."+var_data.name
                raise RuntimeError("Method constraintNode in class "
                                   "ScenarioTree encountered a constraint "
                                   "with variable %s "
                                   "that does not appear to be assigned to "
                                   "any node in the scenario tree" % full_name)

            var_node_index = self._node_list.index(var_node)

            if var_node_index > deepest_node_index:
                deepest_node_index = var_node_index
                deepest_node = var_node

        return deepest_node

class ScenarioTreeBundle(object):

    def __init__(self):

        self._name = None
        self._scenario_names = []
        # This is a compressed scenario tree, just for the bundle.
        self._scenario_tree = None
        # the absolute probability of scenarios associated with this
        # node in the scenario tree.
        self._probability = 0.0

    @property
    def name(self):
        return self._name

    @property
    def scenario_names(self):
        return self._scenario_names

    @property
    def scenario_tree(self):
        return self._scenario_tree

    @property
    def probability(self):
        return self._probability

class ScenarioTree(object):

    # a utility to construct scenario bundles.
    def _construct_scenario_bundles(self, bundles):

        for bundle_name in bundles:

            scenario_list = []
            bundle_probability = 0.0
            for scenario_name in bundles[bundle_name]:
                scenario_list.append(scenario_name)
                bundle_probability += \
                    self._scenario_map[scenario_name].probability

            scenario_tree_for_bundle = self.make_compressed(scenario_list,
                                                            normalize=True)

            scenario_tree_for_bundle.validate()

            new_bundle = ScenarioTreeBundle()
            new_bundle._name = bundle_name
            new_bundle._scenario_names = scenario_list
            new_bundle._scenario_tree = scenario_tree_for_bundle
            new_bundle._probability = bundle_probability

            self._scenario_bundles.append(new_bundle)
            self._scenario_bundle_map[new_bundle.name] = new_bundle

    #
    # a utility to construct the stage objects for this scenario tree.
    # operates strictly by side effects, initializing the self
    # _stages and _stage_map attributes.
    #

    def _construct_stages(self,
                          stage_names,
                          stage_variable_names,
                          stage_cost_variable_names,
                          stage_derived_variable_names):

        # construct the stage objects, which will leave them
        # largely uninitialized - no variable information, in particular.
        for stage_name in stage_names:

            new_stage = ScenarioTreeStage()
            new_stage._name = stage_name
            new_stage._scenario_tree = self

            for variable_string in stage_variable_names[stage_name]:
                if isVariableNameIndexed(variable_string):
                    variable_name, match_template = \
                        extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_stage._variable_templates:
                    new_stage._variable_templates[variable_name] = []
                new_stage._variable_templates[variable_name].append(match_template)

            # not all stages have derived variables defined
            if stage_name in stage_derived_variable_names:
                for variable_string in stage_derived_variable_names[stage_name]:
                    if isVariableNameIndexed(variable_string):
                        variable_name, match_template = \
                            extractVariableNameAndIndex(variable_string)
                    else:
                        variable_name = variable_string
                        match_template = ""
                    if variable_name not in new_stage._derived_variable_templates:
                        new_stage._derived_variable_templates[variable_name] = []
                    new_stage._derived_variable_templates[variable_name].append(match_template)

            # de-reference is required to access the parameter value
            # TBD March 2020: make it so the stages always know their cost names.
            # dlw March 2020: when coming from NetworkX, we don't know these yet!!
            cost_variable_string = stage_cost_variable_names[stage_name].value
            if cost_variable_string is not None:
                if isVariableNameIndexed(cost_variable_string):
                    cost_variable_name, cost_variable_index = \
                        extractVariableNameAndIndex(cost_variable_string)
                else:
                    cost_variable_name = cost_variable_string
                    cost_variable_index = None
                new_stage._cost_variable = (cost_variable_name, cost_variable_index)

            self._stages.append(new_stage)
            self._stage_map[stage_name] = new_stage

    """ Constructor
        Arguments:
            scenarioinstance     - the reference (deterministic) scenario instance.
            scenariotreeinstance - the pyomo model specifying all scenario tree (text) data.
            scenariobundlelist   - a list of scenario names to retain, i.e., cull the rest to create a reduced tree!
    """
    def __init__(self,
                 scenariotreeinstance=None,
                 scenariobundlelist=None):

        # some arbitrary identifier
        self._name = None

        # should be called once for each variable blended across a node
        #self._id_labeler = CounterLabeler()
        self._id_labeler = _CUIDLabeler()

        #
        # the core objects defining the scenario tree.
        #

        # collection of ScenarioTreeNodes
        self._tree_nodes = []
        # collection of ScenarioTreeStages - assumed to be in
        # time-order. the set (provided by the user) itself *must* be
        # ordered.
        self._stages = []
        # collection of Scenarios
        self._scenarios = []
        # collection of ScenarioTreeBundles
        self._scenario_bundles = []

        # dictionaries for the above.
        self._tree_node_map = {}
        self._stage_map = {}
        self._scenario_map = {}
        self._scenario_bundle_map = {}

        # a boolean indicating how data for scenario instances is specified.
        # possibly belongs elsewhere, e.g., in the PH algorithm.
        self._scenario_based_data = None

        if scenariotreeinstance is None:
            assert scenariobundlelist is None
            return

        node_ids = scenariotreeinstance.Nodes
        node_child_ids = scenariotreeinstance.Children
        node_stage_ids = scenariotreeinstance.NodeStage
        node_probability_map = scenariotreeinstance.ConditionalProbability
        stage_ids = scenariotreeinstance.Stages
        stage_variable_ids = scenariotreeinstance.StageVariables
        node_variable_ids = scenariotreeinstance.NodeVariables
        stage_cost_variable_ids = scenariotreeinstance.StageCost
        node_cost_variable_ids = scenariotreeinstance.NodeCost
        if any(scenariotreeinstance.StageCostVariable[i].value is not None
               for i in scenariotreeinstance.StageCostVariable):
            logger.warning("DEPRECATED: The 'StageCostVariable' scenario tree "
                           "model parameter has been renamed to 'StageCost'. "
                           "Please update your scenario tree structure model.")
            if any(stage_cost_variable_ids[i].value is not None
                   for i in stage_cost_variable_ids):
                raise ValueError("The 'StageCostVariable' and 'StageCost' "
                                 "parameters can not both be used on a scenario "
                                 "tree structure model.")
            else:
                stage_cost_variable_ids = scenariotreeinstance.StageCostVariable

        if any(stage_cost_variable_ids[i].value is not None
               for i in stage_cost_variable_ids) and \
           any(node_cost_variable_ids[i].value is not None
               for i in node_cost_variable_ids):
            raise ValueError(
                "The 'StageCost' and 'NodeCost' parameters "
                "can not both be used on a scenario tree "
                "structure model.")
        stage_derived_variable_ids = scenariotreeinstance.StageDerivedVariables
        node_derived_variable_ids = scenariotreeinstance.NodeDerivedVariables
        scenario_ids = scenariotreeinstance.Scenarios
        scenario_leaf_ids = scenariotreeinstance.ScenarioLeafNode
        scenario_based_data = scenariotreeinstance.ScenarioBasedData

        # save the method for instance data storage.
        self._scenario_based_data = scenario_based_data()

        # the input stages must be ordered, for both output purposes
        # and knowledge of the final stage.
        if not stage_ids.isordered():
            raise ValueError(
                "An ordered set of stage IDs must be supplied in "
                "the ScenarioTree constructor")

        for node_id in node_ids:
            node_stage_id = node_stage_ids[node_id].value
            if node_stage_id != stage_ids.last():
                if (len(stage_variable_ids[node_stage_id]) == 0) and \
                   (len(node_variable_ids[node_id]) == 0):
                    raise ValueError(
                        "Scenario tree node %s, belonging to stage %s, "
                        "has not been declared with any variables. "
                        "To fix this error, make sure that one of "
                        "the sets StageVariables[%s] or NodeVariables[%s] "
                        "is declared with at least one variable string "
                        "template (e.g., x, x[*]) on the scenario tree "
                        "or in ScenarioStructure.dat."
                        % (node_id, node_stage_id, node_stage_id, node_id))

        #
        # construct the actual tree objects
        #

        # construct the stage objects w/o any linkages first; link them up
        # with tree nodes after these have been fully constructed.
        self._construct_stages(stage_ids,
                               stage_variable_ids,
                               stage_cost_variable_ids,
                               stage_derived_variable_ids)

        # construct the tree node objects themselves in a first pass,
        # and then link them up in a second pass to form the tree.
        # can't do a single pass because the objects may not exist.
        for tree_node_name in node_ids:

            if tree_node_name not in node_stage_ids:
                raise ValueError("No stage is assigned to tree node=%s"
                                 % (tree_node_name))

            stage_name = value(node_stage_ids[tree_node_name])
            if stage_name not in self._stage_map:
                raise ValueError("Unknown stage=%s assigned to tree node=%s"
                                 % (stage_name, tree_node_name))

            node_stage = self._stage_map[stage_name]
            new_tree_node = ScenarioTreeNode(
                tree_node_name,
                value(node_probability_map[tree_node_name]),
                node_stage)

            # extract the node variable match templates
            for variable_string in node_variable_ids[tree_node_name]:
                if isVariableNameIndexed(variable_string):
                    variable_name, match_template = \
                        extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_tree_node._variable_templates:
                    new_tree_node._variable_templates[variable_name] = []
                new_tree_node._variable_templates[variable_name].append(match_template)

            cost_variable_string = node_cost_variable_ids[tree_node_name].value
            if cost_variable_string is not None:
                assert node_stage._cost_variable is None
                if isVariableNameIndexed(cost_variable_string):
                    cost_variable_name, cost_variable_index = \
                        extractVariableNameAndIndex(cost_variable_string)
                else:
                    cost_variable_name = cost_variable_string
                    cost_variable_index = None
            else:
                assert node_stage._cost_variable is not None
                cost_variable_name, cost_variable_index = \
                    node_stage._cost_variable
            new_tree_node._cost_variable = (cost_variable_name, cost_variable_index)

            # extract the node derived variable match templates
            for variable_string in node_derived_variable_ids[tree_node_name]:
                if isVariableNameIndexed(variable_string):
                    variable_name, match_template = \
                        extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_tree_node._derived_variable_templates:
                    new_tree_node._derived_variable_templates[variable_name] = []
                new_tree_node._derived_variable_templates[variable_name].append(match_template)

            self._tree_nodes.append(new_tree_node)
            self._tree_node_map[tree_node_name] = new_tree_node
            self._stage_map[stage_name]._tree_nodes.append(new_tree_node)

        # link up the tree nodes objects based on the child id sets.
        for this_node in self._tree_nodes:
            this_node._children = []
            # otherwise, you're at a leaf and all is well.
            if this_node.name in node_child_ids:
                child_ids = node_child_ids[this_node.name]
                for child_id in child_ids:
                    if child_id in self._tree_node_map:
                        child_node = self._tree_node_map[child_id]
                        this_node._children.append(child_node)
                        if child_node._parent is None:
                            child_node._parent = this_node
                        else:
                            raise ValueError(
                                "Multiple parents specified for tree node=%s; "
                                "existing parent node=%s; conflicting parent "
                                "node=%s"
                                % (child_id,
                                   child_node._parent.name,
                                   this_node.name))
                    else:
                        raise ValueError("Unknown child tree node=%s specified "
                                         "for tree node=%s"
                                         % (child_id, this_node.name))

        # at this point, the scenario tree nodes and the stages are set - no
        # two-pass logic necessary when constructing scenarios.
        for scenario_name in scenario_ids:
            new_scenario = Scenario()
            new_scenario._name = scenario_name

            if scenario_name not in scenario_leaf_ids:
                raise ValueError("No leaf tree node specified for scenario=%s"
                                 % (scenario_name))
            else:
                scenario_leaf_node_name = value(scenario_leaf_ids[scenario_name])
                if scenario_leaf_node_name not in self._tree_node_map:
                    raise ValueError("Uknown tree node=%s specified as leaf "
                                     "of scenario=%s" %
                                     (scenario_leaf_node_name, scenario_name))
                else:
                    new_scenario._leaf_node = \
                        self._tree_node_map[scenario_leaf_node_name]

            current_node = new_scenario._leaf_node
            while current_node is not None:
                new_scenario._node_list.append(current_node)
                # links the scenarios to the nodes to enforce
                # necessary non-anticipativity
                current_node._scenarios.append(new_scenario)
                current_node = current_node._parent
            new_scenario._node_list.reverse()
            # This now loops root -> leaf
            probability = 1.0
            for current_node in new_scenario._node_list:
                probability *= current_node._conditional_probability
                # NOTE: The line placement below is a little weird, in that
                #       it is embedded in a scenario loop - so the probabilities
                #       for some nodes will be redundantly computed. But this works.
                current_node._probability = probability

                new_scenario._stage_costs[current_node.stage.name] = None
                new_scenario._x[current_node.name] = {}
                new_scenario._w[current_node.name] = {}
                new_scenario._rho[current_node.name] = {}
                new_scenario._fixed[current_node.name] = set()
                new_scenario._stale[current_node.name] = set()

            new_scenario._probability = probability

            self._scenarios.append(new_scenario)
            self._scenario_map[scenario_name] = new_scenario

        # for output purposes, it is useful to known the maximal
        # length of identifiers in the scenario tree for any
        # particular category. I'm building these up incrementally, as
        # they are needed. 0 indicates unassigned.
        self._max_scenario_id_length = 0

        # does the actual traversal to populate the members.
        self.computeIdentifierMaxLengths()

        # if a sub-bundle of scenarios has been specified, mark the
        # active scenario tree components and compress the tree.
        if scenariobundlelist is not None:
            self.compress(scenariobundlelist)

        # NEW SCENARIO BUNDLING STARTS HERE
        if value(scenariotreeinstance.Bundling[None]):
            bundles = OrderedDict()
            for bundle_name in scenariotreeinstance.Bundles:
                bundles[bundle_name] = \
                    list(scenariotreeinstance.BundleScenarios[bundle_name])
            self._construct_scenario_bundles(bundles)

    @property
    def scenarios(self):
        return self._scenarios

    @property
    def bundles(self):
        return self._scenario_bundles

    @property
    def subproblems(self):
        if self.contains_bundles():
            return self._scenario_bundles
        else:
            return self._scenarios

    @property
    def stages(self):
        return self._stages

    @property
    def nodes(self):
        return self._tree_nodes

    def is_bundle(self, object_name):
        return object_name in self._scenario_bundle_map

    def is_scenario(self, object_name):
        return object_name in self._scenario_map

    #
    # Updates the minimum, maximum, and average for all nodes
    # on this tree
    #
    def updateNodeStatistics(self):
        for tree_node in self._tree_nodes:
            tree_node.updateNodeStatistics()

    #
    # populate those portions of the scenario tree and associated
    # stages and tree nodes that reference the scenario instances
    # associated with the tree.
    #

    def linkInInstances(self,
                        scenario_instance_map,
                        objective_sense=None,
                        create_variable_ids=True,
                        master_scenario_tree=None,
                        initialize_solution_data=True):

        if objective_sense not in (minimize, maximize, None):
            raise ValueError(
                "Invalid value (%r) for objective sense given to "
                "the linkInInstances method. Choices are: "
                "[minimize, maximize, None]" % (objective_sense))

        if create_variable_ids and \
           (master_scenario_tree is not None):
            raise RuntimeError(
                "The linkInInstances method of ScenarioTree objects "
                "cannot be invoked with both create_variable_ids=True "
                "and master_scenario_tree!=None")

        # propagate the scenario instances to the scenario tree object
        # structure.
        # NOTE: The input scenario instances may be a super-set of the
        #       set of Scenario objects for this ScenarioTree.
        scenario_names = sorted(scenario_instance_map)
        master_has_instance = {}
        for scenario_name in scenario_names:
            scenario_instance = scenario_instance_map[scenario_name]
            if self.contains_scenario(scenario_name):
                master_has_instance[scenario_name] = False
                if master_scenario_tree is not None:
                    master_scenario = \
                        master_scenario_tree.get_scenario(scenario_name)
                    if master_scenario._instance is not None:
                        master_has_instance[scenario_name] = True
                _scenario = self.get_scenario(scenario_name)
                _scenario._instance = scenario_instance

        # link the scenario tree object structures to the instance components.
        self.populateVariableIndicesAndValues(
            create_variable_ids=create_variable_ids,
            master_scenario_tree=master_scenario_tree,
            initialize_solution_data=initialize_solution_data)

        # create the scenario cost expression to be used for the objective
        for scenario_name in scenario_names:

            scenario_instance = scenario_instance_map[scenario_name]

            if self.contains_scenario(scenario_name):
                scenario = self.get_scenario(scenario_name)

                if master_has_instance[scenario_name]:
                    master_scenario = \
                        master_scenario_tree.get_scenario(scenario_name)
                    scenario._instance_cost_expression = \
                        master_scenario._instance_cost_expression
                    scenario._instance_objective = \
                        master_scenario._instance_objective
                    scenario._objective_sense =\
                        master_scenario._objective_sense
                    scenario._objective_name = master_scenario
                    continue

                user_objective = find_active_objective(scenario_instance,
                                                       safety_checks=True)
                if objective_sense is None:
                    if user_objective is None:
                        raise RuntimeError(
                            "An active Objective could not be found on "
                            "instance for scenario %s." % (scenario_name))
                    cost_expr_name = "_PySP_UserCostExpression"
                    cost_expr = Expression(name=cost_expr_name,
                                           initialize=user_objective.expr)
                    scenario_instance.add_component(cost_expr_name, cost_expr)
                    scenario._instance_cost_expression = cost_expr

                    # We have wrapped the original objective expression
                    # in an Expression object so that other code can
                    # augment the objective with other terms without
                    # modifying the original objective expression. This
                    # also allows us to easily reset the objective to its
                    # original form
                    user_objective.expr = cost_expr
                    scenario._instance_objective = user_objective
                    scenario._objective_sense = user_objective.sense
                    scenario._objective_name = \
                        scenario._instance_objective.name

                else:

                    if user_objective is not None:
                        print("*** Active Objective \'%s\' on scenario "
                              "instance \'%s\' will not be used. ***"
                              % (user_objective.name, scenario_name))
                        user_objective.deactivate()

                    cost = 0.0
                    for node in scenario.node_list:
                        stage_cost_var = \
                            scenario_instance.\
                            find_component(node._cost_variable[0])\
                            [node._cost_variable[1]]
                        cost += stage_cost_var
                    cost_expr_name = "_PySP_CostExpression"
                    cost_expr = Expression(name=cost_expr_name,
                                           initialize=cost)
                    scenario_instance.add_component(cost_expr_name,cost_expr)
                    scenario._instance_cost_expression = cost_expr

                    cost_obj_name = "_PySP_CostObjective"
                    cost_obj = Objective(name=cost_obj_name,
                                         expr=cost_expr,
                                         sense=objective_sense)
                    scenario_instance.add_component(cost_obj_name,cost_obj)
                    scenario._instance_objective = cost_obj
                    scenario._instance_original_objective_object = \
                        user_objective
                    scenario._objective_sense = objective_sense
                    scenario._objective_name = \
                        scenario._instance_objective.name

    #
    # compute the set of variable indices being blended at each
    # node. this can't be done until all of the scenario instances are
    # available, as different scenarios can have different index sets.
    #

    def populateVariableIndicesAndValues(self,
                                         create_variable_ids=True,
                                         master_scenario_tree=None,
                                         initialize_solution_data=True):
        if (create_variable_ids == True) and \
           (master_scenario_tree != None):
            raise RuntimeError(
                "The populateVariableIndicesAndValues method of "
                "ScenarioTree objects cannot be invoked with both "
                "create_variable_ids=True and master_scenario_tree!=None")

        labeler = None
        if create_variable_ids:
            labeler = self._id_labeler
            for scenario in self.scenarios:
                labeler.update_cache(scenario.instance)

        for stage in self._stages:
            tree_node_list = sorted(stage._tree_nodes, key=lambda x: x.name)
            for tree_node in tree_node_list:
                name_index_to_id_map = None
                if master_scenario_tree is not None:
                    name_index_to_id_map = master_scenario_tree.\
                                           get_node(tree_node.name).\
                                           _name_index_to_id
                tree_node.populateVariableIndicesAndValues(
                    id_labeler=labeler,
                    name_index_to_id_map=name_index_to_id_map,
                    initialize_solution_data=initialize_solution_data)

        if labeler is not None:
            labeler.clear_cache()

    #
    # is the indicated scenario / bundle in the tree?
    #

    def contains_scenario(self, name):
        return name in self._scenario_map

    def contains_bundles(self):
        return len(self._scenario_bundle_map) > 0

    def contains_bundle(self, name):
        return name in self._scenario_bundle_map

    #
    # get the scenario / bundle object from the tree.
    #

    def get_scenario(self, name):
        return self._scenario_map[name]

    def get_bundle(self, name):
        return self._scenario_bundle_map[name]

    def get_subproblem(self, name):
        if self.contains_bundles():
            return self._scenario_bundle_map[name]
        else:
            return self._scenario_map[name]

    def get_scenario_bundle(self, name):
        if not self.contains_bundles():
            return None
        else:
            return self._scenario_bundle_map[name]

    # there are many contexts where manipulators of a scenario
    # tree simply need an arbitrary scenario to proceed...
    def get_arbitrary_scenario(self):
        return self._scenarios[0]

    def contains_node(self, name):
        return name in self._tree_node_map

    #
    # get the scenario tree node object from the tree
    #
    def get_node(self, name):
        return self._tree_node_map[name]

    #
    # utility for compressing or culling a scenario tree based on
    # a provided list of scenarios (specified by name) to retain -
    # all non-referenced components are eliminated. this particular
    # method compresses *in-place*, i.e., via direct modification
    # of the scenario tree structure. If normalize=True, all probabilities
    # (and conditional probabilities) are renormalized.
    #

    def compress(self,
                 scenario_bundle_list,
                 normalize=True):

        # scan for and mark all referenced scenarios and
        # tree nodes in the bundle list - all stages will
        # obviously remain.
        try:

            for scenario_name in scenario_bundle_list:

                scenario = self._scenario_map[scenario_name]
                scenario.retain = True

                # chase all nodes comprising this scenario,
                # marking them for retention.
                for node in scenario._node_list:
                    node.retain = True

        except KeyError:
            raise ValueError("Scenario=%s selected for "
                             "bundling not present in "
                             "scenario tree"
                             % (scenario_name))

        # scan for any non-retained scenarios and tree nodes.
        scenarios_to_delete = []
        tree_nodes_to_delete = []
        for scenario in self._scenarios:
            if hasattr(scenario, "retain"):
                delattr(scenario, "retain")
            else:
                scenarios_to_delete.append(scenario)
                del self._scenario_map[scenario.name]

        for tree_node in self._tree_nodes:
            if hasattr(tree_node, "retain"):
                delattr(tree_node, "retain")
            else:
                tree_nodes_to_delete.append(tree_node)
                del self._tree_node_map[tree_node.name]

        # JPW does not claim the following routines are
        # the most efficient. rather, they get the job
        # done while avoiding serious issues with
        # attempting to remove elements from a list that
        # you are iterating over.

        # delete all references to unmarked scenarios
        # and child tree nodes in the scenario tree node
        # structures.
        for tree_node in self._tree_nodes:
            for scenario in scenarios_to_delete:
                if scenario in tree_node._scenarios:
                    tree_node._scenarios.remove(scenario)
            for node_to_delete in tree_nodes_to_delete:
                if node_to_delete in tree_node._children:
                    tree_node._children.remove(node_to_delete)

        # delete all references to unmarked tree nodes
        # in the scenario tree stage structures.
        for stage in self._stages:
            for tree_node in tree_nodes_to_delete:
                if tree_node in stage._tree_nodes:
                    stage._tree_nodes.remove(tree_node)

        # delete all unreferenced entries from the core scenario
        # tree data structures.
        for scenario in scenarios_to_delete:
            self._scenarios.remove(scenario)
        for tree_node in tree_nodes_to_delete:
            self._tree_nodes.remove(tree_node)

        #
        # Handle re-normalization of probabilities if requested
        #
        if normalize:

            # re-normalize the conditional probabilities of the
            # children at each tree node (leaf-to-root stage order).
            for stage in reversed(self._stages[:-1]):

                for tree_node in stage._tree_nodes:
                    norm_factor = sum(child_tree_node._conditional_probability
                                      for child_tree_node
                                      in tree_node._children)
                    # the user may specify that the probability of a
                    # scenario is 0.0, and while odd, we should allow the
                    # edge case.
                    if norm_factor == 0.0:
                        for child_tree_node in tree_node._children:
                            child_tree_node._conditional_probability = 0.0
                    else:
                        for child_tree_node in tree_node._children:
                            child_tree_node._conditional_probability /= norm_factor

            # update absolute probabilities (root-to-leaf stage order)
            for stage in self._stages[1:]:
                for tree_node in stage._tree_nodes:
                    tree_node._probability = \
                        tree_node._parent._probability * \
                        tree_node._conditional_probability

            # update scenario probabilities
            for scenario in self._scenarios:
                scenario._probability = \
                    scenario._leaf_node._probability

        # now that we've culled the scenarios, cull the bundles. do
        # this in two passes. in the first pass, we identify the names
        # of bundles to delete, by looking for bundles with deleted
        # scenarios. in the second pass, we delete the bundles from
        # the scenario tree, and normalize the probabilities of the
        # remaining bundles.

        # indices of the objects in the scenario tree bundle list
        bundles_to_delete = []
        for i in xrange(0,len(self._scenario_bundles)):
            scenario_bundle = self._scenario_bundles[i]
            for scenario_name in scenario_bundle._scenario_names:
                if scenario_name not in self._scenario_map:
                    bundles_to_delete.append(i)
                    break
        bundles_to_delete.reverse()
        for i in bundles_to_delete:
            deleted_bundle = self._scenario_bundles.pop(i)
            del self._scenario_bundle_map[deleted_bundle.name]

        sum_bundle_probabilities = \
            sum(bundle._probability for bundle in self._scenario_bundles)
        for bundle in self._scenario_bundles:
            bundle._probability /= sum_bundle_probabilities

    #
    # Returns a compressed tree using operations on the order of the
    # number of nodes in the compressed tree rather than the number of
    # nodes in the full tree (this method is more efficient than in-place
    # compression). If normalize=True, all probabilities
    # (and conditional probabilities) are renormalized.
    #
    # *** Bundles are ignored. The compressed tree will not have them ***
    #
    def make_compressed(self,
                        scenario_bundle_list,
                        normalize=False):

        compressed_tree = ScenarioTree()
        compressed_tree._scenario_based_data = self._scenario_based_data
        #
        # Copy Stage Data
        #
        for stage in self._stages:
            # copy everything but the list of tree nodes
            # and the reference to the scenario tree
            compressed_tree_stage = ScenarioTreeStage()
            compressed_tree_stage._name = stage.name
            compressed_tree_stage._variable_templates = copy.deepcopy(stage._variable_templates)
            compressed_tree_stage._derived_variable_templates = \
                copy.deepcopy(stage._derived_variable_templates)
            compressed_tree_stage._cost_variable = copy.deepcopy(stage._cost_variable)
            # add the stage object to the compressed tree
            compressed_tree._stages.append(compressed_tree_stage)
            compressed_tree._stages[-1]._scenario_tree = compressed_tree

        compressed_tree._stage_map = \
            dict((stage.name, stage) for stage in compressed_tree._stages)

        #
        # Copy Scenario and Node Data
        #
        compressed_tree_root = None
        for scenario_name in scenario_bundle_list:
            full_tree_scenario = self.get_scenario(scenario_name)

            compressed_tree_scenario = Scenario()
            compressed_tree_scenario._name = full_tree_scenario.name
            compressed_tree_scenario._probability = full_tree_scenario._probability
            compressed_tree._scenarios.append(compressed_tree_scenario)

            full_tree_node = full_tree_scenario._leaf_node
            ### copy the node
            compressed_tree_node = ScenarioTreeNode(
                full_tree_node.name,
                full_tree_node._conditional_probability,
                compressed_tree._stage_map[full_tree_node._stage.name])
            compressed_tree_node._variable_templates = \
                copy.deepcopy(full_tree_node._variable_templates)
            compressed_tree_node._derived_variable_templates = \
                copy.deepcopy(full_tree_node._derived_variable_templates)
            compressed_tree_node._scenarios.append(compressed_tree_scenario)
            compressed_tree_node._stage._tree_nodes.append(compressed_tree_node)
            compressed_tree_node._probability = full_tree_node._probability
            compressed_tree_node._cost_variable = full_tree_node._cost_variable
            ###

            compressed_tree_scenario._node_list.append(compressed_tree_node)
            compressed_tree_scenario._leaf_node = compressed_tree_node
            compressed_tree._tree_nodes.append(compressed_tree_node)
            compressed_tree._tree_node_map[compressed_tree_node.name] = \
                compressed_tree_node

            previous_compressed_tree_node = compressed_tree_node
            full_tree_node = full_tree_node._parent
            while full_tree_node.name not in compressed_tree._tree_node_map:

                ### copy the node
                compressed_tree_node = ScenarioTreeNode(
                    full_tree_node.name,
                    full_tree_node._conditional_probability,
                    compressed_tree._stage_map[full_tree_node.stage.name])
                compressed_tree_node._variable_templates = \
                    copy.deepcopy(full_tree_node._variable_templates)
                compressed_tree_node._derived_variable_templates = \
                    copy.deepcopy(full_tree_node._derived_variable_templates)
                compressed_tree_node._probability = full_tree_node._probability
                compressed_tree_node._cost_variable = full_tree_node._cost_variable
                compressed_tree_node._scenarios.append(compressed_tree_scenario)
                compressed_tree_node._stage._tree_nodes.append(compressed_tree_node)
                ###

                compressed_tree_scenario._node_list.append(compressed_tree_node)
                compressed_tree._tree_nodes.append(compressed_tree_node)
                compressed_tree._tree_node_map[compressed_tree_node.name] = \
                    compressed_tree_node
                previous_compressed_tree_node._parent = compressed_tree_node
                compressed_tree_node._children.append(previous_compressed_tree_node)
                previous_compressed_tree_node = compressed_tree_node

                full_tree_node = full_tree_node._parent
                if full_tree_node is None:
                    compressed_tree_root = compressed_tree_node
                    break

            # traverse the remaining nodes up to the root and update the
            # tree structure elements
            if full_tree_node is not None:
                compressed_tree_node = \
                    compressed_tree._tree_node_map[full_tree_node.name]
                previous_compressed_tree_node._parent = compressed_tree_node
                compressed_tree_node._scenarios.append(compressed_tree_scenario)
                compressed_tree_node._children.append(previous_compressed_tree_node)
                compressed_tree_scenario._node_list.append(compressed_tree_node)

                compressed_tree_node = compressed_tree_node._parent
                while compressed_tree_node is not None:
                    compressed_tree_scenario._node_list.append(compressed_tree_node)
                    compressed_tree_node._scenarios.append(compressed_tree_scenario)
                    compressed_tree_node = compressed_tree_node._parent

            # makes sure this list is in root to leaf order
            compressed_tree_scenario._node_list.reverse()
            assert compressed_tree_scenario._node_list[-1] is \
                compressed_tree_scenario._leaf_node
            assert compressed_tree_scenario._node_list[0] is \
                compressed_tree_root

            # initialize solution related dictionaries
            for compressed_tree_node in compressed_tree_scenario._node_list:
                compressed_tree_scenario._stage_costs[compressed_tree_node._stage.name] = None
                compressed_tree_scenario._x[compressed_tree_node.name] = {}
                compressed_tree_scenario._w[compressed_tree_node.name] = {}
                compressed_tree_scenario._rho[compressed_tree_node.name] = {}
                compressed_tree_scenario._fixed[compressed_tree_node.name] = set()
                compressed_tree_scenario._stale[compressed_tree_node.name] = set()

        compressed_tree._scenario_map = \
            dict((scenario.name, scenario) for scenario in compressed_tree._scenarios)

        #
        # Handle re-normalization of probabilities if requested
        #
        if normalize:

            # update conditional probabilities (leaf-to-root stage order)
            for compressed_tree_stage in reversed(compressed_tree._stages[:-1]):

                for compressed_tree_node in compressed_tree_stage._tree_nodes:
                    norm_factor = \
                        sum(compressed_tree_child_node._conditional_probability
                            for compressed_tree_child_node
                            in compressed_tree_node._children)
                    # the user may specify that the probability of a
                    # scenario is 0.0, and while odd, we should allow the
                    # edge case.
                    if norm_factor == 0.0:
                        for compressed_tree_child_node in \
                               compressed_tree_node._children:
                            compressed_tree_child_node._conditional_probability = 0.0

                    else:
                        for compressed_tree_child_node in \
                               compressed_tree_node._children:
                            compressed_tree_child_node.\
                                _conditional_probability /= norm_factor

            assert abs(compressed_tree_root._probability - 1.0) < 1e-5
            assert abs(compressed_tree_root._conditional_probability - 1.0) < 1e-5

            # update absolute probabilities (root-to-leaf stage order)
            for compressed_tree_stage in compressed_tree._stages[1:]:
                for compressed_tree_node in compressed_tree_stage._tree_nodes:
                    compressed_tree_node._probability = \
                            compressed_tree_node._parent._probability * \
                            compressed_tree_node._conditional_probability

            # update scenario probabilities
            for compressed_tree_scenario in compressed_tree._scenarios:
                compressed_tree_scenario._probability = \
                    compressed_tree_scenario._leaf_node._probability

        return compressed_tree

    #
    # Adds a bundle to this scenario tree by calling make compressed
    # with normalize=True
    # Returns a compressed tree using operations on the order of the
    # number of nodes in the compressed tree rather than the number of
    # nodes in the full tree (this method is more efficient than in-place
    # compression). If normalize=True, all probabilities
    # (and conditional probabilities) are renormalized.
    #
    #
    def add_bundle(self, name, scenario_bundle_list):

        if name in self._scenario_bundle_map:
            raise ValueError("Cannot add a new bundle with name '%s', a bundle "
                             "with that name already exists." % (name))

        bundle_scenario_tree = self.make_compressed(scenario_bundle_list,
                                                    normalize=True)
        bundle = ScenarioTreeBundle()
        bundle._name = name
        bundle._scenario_names = scenario_bundle_list
        bundle._scenario_tree = bundle_scenario_tree
        # make sure this is computed with the un-normalized bundle scenarios
        bundle._probability = sum(self._scenario_map[scenario_name]._probability
                                  for scenario_name in scenario_bundle_list)

        self._scenario_bundle_map[name] = bundle
        self._scenario_bundles.append(bundle)

    def remove_bundle(self, name):

        if name not in self._scenario_bundle_map:
            raise KeyError("Cannot remove bundle with name '%s', no bundle "
                           "with that name exists." % (name))
        bundle = self._scenario_bundle_map[name]
        del self._scenario_bundle_map[name]
        self._scenario_bundles.remove(bundle)

    #
    # utility for automatically selecting a proportion of scenarios from the
    # tree to retain, eliminating the rest.
    #

    def downsample(self, fraction_to_retain, random_seed, verbose=False):

        random_state = random.getstate()
        random.seed(random_seed)
        try:
            number_to_retain = \
                max(int(round(float(len(self._scenarios)*fraction_to_retain))), 1)
            random_list=random.sample(range(len(self._scenarios)), number_to_retain)

            scenario_bundle_list = []
            for i in xrange(number_to_retain):
                scenario_bundle_list.append(self._scenarios[random_list[i]].name)

            if verbose:
                print("Downsampling scenario tree - retained %s "
                      "scenarios: %s"
                      % (len(scenario_bundle_list),
                         str(scenario_bundle_list)))

            self.compress(scenario_bundle_list) # do the downsampling
        finally:
            random.setstate(random_state)


    #
    # returns the root node of the scenario tree
    #

    def findRootNode(self):

        for tree_node in self._tree_nodes:
            if tree_node._parent is None:
                return tree_node
        return None

    #
    # a utility function to compute, based on the current scenario tree content,
    # the maximal length of identifiers in various categories.
    #

    def computeIdentifierMaxLengths(self):

        self._max_scenario_id_length = 0
        for scenario in self._scenarios:
            if len(str(scenario.name)) > self._max_scenario_id_length:
                self._max_scenario_id_length = len(str(scenario.name))

    #
    # a utility function to (partially, at the moment) validate a scenario tree
    #

    def validate(self):

        # for any node, the sum of conditional probabilities of the children should sum to 1.
        for tree_node in self._tree_nodes:
            sum_probabilities = 0.0
            if len(tree_node._children) > 0:
                for child in tree_node._children:
                    sum_probabilities += child._conditional_probability
                if abs(1.0 - sum_probabilities) > 0.000001:
                    raise ValueError("ScenarioTree validation failed. "
                                     "Reason: child conditional "
                                     "probabilities for tree node=%s "
                                     " sum to %s"
                                     % (tree_node.name,
                                        sum_probabilities))

        # ensure that there is only one root node in the tree
        num_roots = 0
        root_ids = []
        for tree_node in self._tree_nodes:
            if tree_node._parent is None:
                num_roots += 1
                root_ids.append(tree_node.name)

        if num_roots != 1:
            raise ValueError("ScenarioTree validation failed. "
                             "Reason: illegal set of root "
                             "nodes detected: " + str(root_ids))

        # there must be at least one scenario passing through each tree node.
        for tree_node in self._tree_nodes:
            if len(tree_node._scenarios) == 0:
                raise ValueError("ScenarioTree validation failed. "
                                 "Reason: there are no scenarios "
                                 "associated with tree node=%s"
                                 % (tree_node.name))
                return False

        return True

    #
    # copies the parameter values stored in any tree node _averages attribute
    # into any tree node _solution attribute - only for active variable values.
    #

    def snapshotSolutionFromAverages(self):

        for tree_node in self._tree_nodes:

            tree_node.snapshotSolutionFromAverages()

    #
    # assigns the variable values at each tree node based on the input
    # instances.
    #

    # Note: Trying to work this function out of the code. The only
    #       solution we should get used to working with is that stored
    #       on the scenario objects
    def XsnapshotSolutionFromInstances(self):

        for tree_node in self._tree_nodes:
            tree_node.snapshotSolutionFromInstances()

    def pullScenarioSolutionsFromInstances(self):

        for scenario in self._scenarios:
            scenario.update_solution_from_instance()

    def snapshotSolutionFromScenarios(self):
        for tree_node in self._tree_nodes:
            tree_node.snapshotSolutionFromScenarios()

    def create_random_bundles(self,
                              num_bundles,
                              random_seed):

        random_state = random.getstate()
        random.seed(random_seed)
        try:
            num_scenarios = len(self._scenarios)

            sequence = list(range(num_scenarios))
            random.shuffle(sequence)

            next_scenario_index = 0

            # this is a hack-ish way to re-initialize the Bundles set of a
            # scenario tree instance, which should already be there
            # (because it is defined in the abstract model).  however, we
            # don't have a "clear" method on a set, so...
            bundle_names = ["Bundle"+str(i)
                            for i in xrange(1, num_bundles+1)]
            bundles = OrderedDict()
            for i in xrange(num_bundles):
                bundles[bundle_names[i]] = []

            scenario_index = 0
            while (scenario_index < num_scenarios):
                for bundle_index in xrange(num_bundles):
                    if (scenario_index == num_scenarios):
                        break
                    bundles[bundle_names[bundle_index]].append(
                        self._scenarios[sequence[scenario_index]].name)
                    scenario_index += 1

            self._construct_scenario_bundles(bundles)
        finally:
            random.setstate(random_state)

    #
    # a utility function to pretty-print the static/non-cost
    # information associated with a scenario tree
    #

    def pprint(self):

        print("Scenario Tree Detail")

        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(iterkeys(self._tree_node_map)):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            if tree_node._conditional_probability is not None:
                print("\tConditional probability=%4.4f" % tree_node._conditional_probability)
            else:
                print("\tConditional probability=" + "***Undefined***")
            print("\tChildren:")
            if len(tree_node._children) > 0:
                for child_node in sorted(tree_node._children, key=lambda x: x._name):
                    print("\t\t%s" % (child_node._name))
            else:
                print("\t\tNone")
            print("\tScenarios:")
            if len(tree_node._scenarios) == 0:
                print("\t\tNone")
            else:
                for scenario in sorted(tree_node._scenarios, key=lambda x: x._name):
                    print("\t\t%s" % (scenario._name))
            if len(tree_node._variable_templates) > 0:
                print("\tVariables: ")
                for variable_name in sorted(iterkeys(tree_node._variable_templates)):
                    match_templates = tree_node._variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            if len(tree_node._derived_variable_templates) > 0:
                print("\tDerived Variables: ")
                for variable_name in sorted(iterkeys(tree_node._derived_variable_templates)):
                    match_templates = tree_node._derived_variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            print("")
        print("----------------------------------------------------")
        print("Stages:")
        for stage_name in sorted(iterkeys(self._stage_map)):
            stage = self._stage_map[stage_name]
            print("\tName=%s" % (stage_name))
            print("\tTree Nodes: ")
            for tree_node in sorted(stage._tree_nodes, key=lambda x: x._name):
                print("\t\t%s" % (tree_node._name))
            if len(stage._variable_templates) > 0:
                print("\tVariables: ")
                for variable_name in sorted(iterkeys(stage._variable_templates)):
                    match_templates = stage._variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            if len(stage._derived_variable_templates) > 0:
                print("\tDerived Variables: ")
                for variable_name in sorted(iterkeys(stage._derived_variable_templates)):
                    match_templates = stage._derived_variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            print("\tCost Variable: ")
            if stage._cost_variable is not None:
                cost_variable_name, cost_variable_index = stage._cost_variable
            else:
                # kind of a hackish way to get around the fact that we are transitioning
                # away from storing the cost_variable identifier on the stages
                cost_variable_name, cost_variable_index = stage.nodes[0]._cost_variable
            if cost_variable_index is None:
                print("\t\t" + cost_variable_name)
            else:
                print("\t\t" + cost_variable_name + indexToString(cost_variable_index))
            print("")
        print("----------------------------------------------------")
        print("Scenarios:")
        for scenario_name in sorted(iterkeys(self._scenario_map)):
            scenario = self._scenario_map[scenario_name]
            print("\tName=%s" % (scenario_name))
            print("\tProbability=%4.4f" % scenario._probability)
            if scenario._leaf_node is None:
                print("\tLeaf node=None")
            else:
                print("\tLeaf node=%s" % (scenario._leaf_node._name))
            print("\tTree node sequence:")
            for tree_node in scenario._node_list:
                print("\t\t%s" % (tree_node._name))
            print("")
        print("----------------------------------------------------")
        if len(self._scenario_bundles) > 0:
            print("Scenario Bundles:")
            for bundle_name in sorted(iterkeys(self._scenario_bundle_map)):
                scenario_bundle = self._scenario_bundle_map[bundle_name]
                print("\tName=%s" % (bundle_name))
                print("\tProbability=%4.4f" % scenario_bundle._probability            )
                sys.stdout.write("\tScenarios:  ")
                for scenario_name in sorted(scenario_bundle._scenario_names):
                    sys.stdout.write(str(scenario_name)+' ')
                sys.stdout.write("\n")
                print("")
            print("----------------------------------------------------")

    #
    # a utility function to pretty-print the solution associated with a scenario tree
    #

    def pprintSolution(self, epsilon=1.0e-5):

        #print("Scenario Tree Solution")
        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(iterkeys(self._tree_node_map)):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            label_printed = False
            if (len(tree_node._stage._variable_templates) > 0) or \
               (len(tree_node._variable_templates) > 0):
                for name in sorted(tree_node._variable_indices):
                    for index in sorted(tree_node._variable_indices[name]):
                        id_ = tree_node._name_index_to_id[name,index]
                        if id_ in tree_node._standard_variable_ids:
                            if not label_printed:
                                print("\tVariables: ")
                                label_printed = True
                            # if a solution has not yet been stored /
                            # snapshotted, then the value won't be in the solution map
                            try:
                                value = tree_node._solution[id_]
                            except KeyError:
                                value = None
                            if (value is not None) and (math.fabs(value) > epsilon):
                                print("\t\t"+name+indexToString(index)+"="+str(value))
            label_printed = False
            if (len(tree_node._stage._derived_variable_templates) > 0) or \
               (len(tree_node._derived_variable_templates) > 0):
                for name in sorted(tree_node._variable_indices):
                    for index in sorted(tree_node._variable_indices[name]):
                        id_ = tree_node._name_index_to_id[name,index]
                        if id_ in tree_node._derived_variable_ids:
                            if not label_printed:
                                print("\tDerived Variables: ")
                                label_printed = True
                            # if a solution has not yet been stored /
                            # snapshotted, then the value won't be in the solution map
                            try:
                                value = tree_node._solution[id_]
                            except KeyError:
                                value = None
                            if (value is not None) and (math.fabs(value) > epsilon):
                                print("\t\t"+name+indexToString(index)+"="+str(value))
            print("")

    #
    # a utility function to pretty-print the cost information associated with a scenario tree
    #

    def pprintCosts(self):

        print("Scenario Tree Costs")
        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(iterkeys(self._tree_node_map)):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            if tree_node._conditional_probability is not None:
                print("\tConditional probability=%4.4f"
                      % tree_node._conditional_probability)
            else:
                print("\tConditional probability=" + "Not Rprted.")
            print("\tChildren:")
            if len(tree_node._children) > 0:
                for child_node in sorted(tree_node._children, key=lambda x: x._name):
                    print("\t\t%s" % (child_node._name))
            else:
                print("\t\tNone")
            print("\tScenarios:")
            if len(tree_node._scenarios) == 0:
                print("\t\tNone")
            else:
                for scenario in sorted(tree_node._scenarios, key=lambda x: x._name):
                    print("\t\t%s" % (scenario._name))
            expected_node_cost = tree_node.computeExpectedNodeCost()
            if expected_node_cost != None:
                print("\tExpected cost of (sub)tree rooted at node=%10.4f"
                      % expected_node_cost)
            else:
                print("\tExpected cost of (sub)tree rooted at node=Not Rprted.")
            print("")

        print("----------------------------------------------------")
        print("Scenarios:")
        print("")
        for scenario_name in sorted(iterkeys(self._scenario_map)):
            scenario = self._scenario_map[scenario_name]

            print("\tName=%s" % (scenario_name))
            print("\tProbability=%4.4f" % scenario._probability)

            if scenario._leaf_node is None:
                print("\tLeaf Node=None")
            else:
                print("\tLeaf Node=%s" % (scenario._leaf_node._name))

            print("\tTree node sequence:")
            for tree_node in scenario._node_list:
                print("\t\t%s" % (tree_node._name))

            aggregate_cost = 0.0
            for stage in self._stages:
                # find the tree node for this scenario, representing this stage.
                tree_node = None
                for node in scenario._node_list:
                    if node._stage == stage:
                        tree_node = node
                        break

                cost_variable_value = scenario._stage_costs[stage._name]

                if cost_variable_value is not None:
                    print("\tStage=%20s     Cost=%10.4f"
                          % (stage._name, cost_variable_value))
                    cost = cost_variable_value
                else:
                    print("\tStage=%20s     Cost=%10s"
                          % (stage._name, "Not Rprted."))
                    cost = 0.0
                aggregate_cost += cost

            print("\tTotal scenario cost=%10.4f" % aggregate_cost)
            print("")
        print("----------------------------------------------------")

    #
    # Save the tree structure in DOT file format
    # Nodes are labeled with absolute probabilities and
    # edges are labeled with conditional probabilities
    #
    def save_to_dot(self, filename):

        def _visit_node(node):
            f.write("%s%s [label=\"%s\"];\n"
                    % (node.name,
                       id(node),
                       str(node.name)+("\n(%.6g)" % (node._probability))))
            for child_node in node._children:
                _visit_node(child_node)
                f.write("%s%s -> %s%s [label=\"%.6g\"];\n"
                        % (node.name,
                           id(node),
                           child_node.name,
                           id(child_node),
                           child_node._conditional_probability))
            if len(node._children) == 0:
                assert len(node._scenarios) == 1
                scenario = node._scenarios[0]
                f.write("%s%s [label=\"%s\"];\n"
                        % (scenario.name,
                           id(scenario),
                           "scenario\n"+str(scenario.name)))
                f.write("%s%s -> %s%s [style=dashed];\n"
                        % (node.name,
                           id(node),
                           scenario.name,
                           id(scenario)))

        with open(filename, 'w') as f:

            f.write("digraph ScenarioTree {\n")
            root_node = self.findRootNode()
            _visit_node(root_node)
            f.write("}\n")
