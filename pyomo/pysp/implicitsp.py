#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ImplicitSP,")

import itertools

import pyomo.core.base.expr
import pyomo.core.base.param
from pyomo.core.base.numvalue import is_fixed, is_constant
from pyomo.core.base.block import (Block,
                                   SortComponents)
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.objective import Objective
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.param import _ParamData
from pyomo.core.base.suffix import ComponentMap

from pyomo.pysp.annotations import (locate_annotations,
                                    PySP_StageCostAnnotation,
                                    PySP_VariableStageAnnotation,
                                    PySP_StochasticDataAnnotation)
from pyomo.pysp.scenariotree.tree_structure import ScenarioTree
from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateAbstractScenarioTreeModel

# TODO: Address the fact that Pyomo variables return
#       numbers when asking for bounds, making it impossible
#       to check if a mutable Param (e.g., stochastic data)
#       appears there.

class ImplicitSP(object):

    @staticmethod
    def _collect_mutable_parameters(exp):
        """
        A helper function for querying a pyomo expression for
        mutable parameters.
        """
        if is_constant(exp) or isinstance(exp, _VarData):
            return {}
        if exp.is_expression():
            ans = {}
            if exp.__class__ is pyomo.core.base.expr._ProductExpression:
                for subexp in exp._numerator:
                    ans.update(ImplicitSP.\
                               _collect_mutable_parameters(subexp))
                for subexp in exp._denominator:
                    ans.update(ImplicitSP.\
                               _collect_mutable_parameters(subexp))
            else:
                # This is fragile: we assume that all other expression
                # objects "play nice" and just use the _args member.
                for subexp in exp._args:
                    ans.update(ImplicitSP.\
                               _collect_mutable_parameters(subexp))
            return ans
        elif isinstance(exp, _ParamData):
            return {id(exp): exp}
        else:
            raise ValueError("Unexpected expression type: "+str(exp))

    @staticmethod
    def _collect_unfixed_variables(exp):
        if is_fixed(exp):
            return {}
        elif exp.is_expression():
            ans = {}
            if exp.__class__ is pyomo.core.base.expr._ProductExpression:
                for subexp in exp._numerator:
                    ans.update(ImplicitSP.\
                               _collect_unfixed_variables(subexp))
                for subexp in exp._denominator:
                    ans.update(ImplicitSP.\
                               _collect_unfixed_variables(subexp))
            else:
                # This is fragile: we assume that all other expression
                # objects "play nice" and just use the _args member.
                for subexp in exp._args:
                    ans.update(ImplicitSP.\
                               _collect_unfixed_variables(subexp))
            return ans
        elif isinstance(exp, _VarData):
            return {id(exp): exp}
        else:
            raise ValueError("Unexpected expression type: "+str(exp))

    def __init__(self, reference_model):

        self.reference_model = None
        self.objective = None
        self.scenario_tree = None

        self.stage_to_variables_map = {}
        self.variable_to_stage_map = {}

        # the set of stochastic data objects
        # (possibly mapped to some distribution)
        self.stochastic_data = None

        # maps between variables and objectives
        self.variable_to_objectives_map = ComponentMap()
        self.objective_to_variables_map = ComponentMap()

        # maps between variables and constraints
        self.variable_to_constraints_map = ComponentMap()
        self.constraint_to_variables_map = ComponentMap()

        # maps between stochastic data and objectives
        self.stochastic_data_to_objectives_map = ComponentMap()
        self.objective_to_stochastic_data_map = ComponentMap()

        # maps between stochastic data and constraints
        self.stochastic_data_to_constraints_map = ComponentMap()
        self.constraint_to_stochastic_data_map = ComponentMap()

        # maps between stochastic data and variables lower and upper bounds
        self.stochastic_data_to_variables_lb_map = ComponentMap()
        self.variable_to_stochastic_data_lb_map = ComponentMap()

        self.stochastic_data_to_variables_ub_map = ComponentMap()
        self.variable_to_stochastic_data_ub_map = ComponentMap()

        if not isinstance(reference_model, Block):
            raise TypeError("reference model input must be a Pyomo model")
        self.reference_model = reference_model

        #
        # Extract stochastic parameters from the PySP_StochasticDataAnnotation object
        #

        self.stochastic_data = \
            self._extract_stochastic_data(self.reference_model)

        #
        # Get the variable stages from the PySP_VariableStageAnnotation object
        #

        (self.stage_to_variables_map,
         self.variable_to_stage_map,
         variable_stage_assignments) = \
            self._map_variable_stages(self.reference_model)

        #
        # Get the stage cost components from the PySP_StageCostAnnotation
        # and generate a dummy single-scenario scenario tree
        #

        self.scenario_tree = \
            self._generate_base_scenario_tree(self.reference_model,
                                              variable_stage_assignments)

        #
        # Extract the locations of variables and stochastic data
        # within the model
        #

        for objcntr, objdata in enumerate(self.reference_model.component_data_objects(
                Objective,
                active=True,
                descend_into=True), 1):

            if objcntr > 1:
                raise ValueError("Reference model can not contain more than one "
                                 "active objective")

            self.objective = objdata
            self.objective_sense = objdata.sense

            obj_params = \
                tuple(self._collect_mutable_parameters(objdata.expr).values())
            self.objective_to_stochastic_data_map[objdata] = []
            for paramdata in obj_params:
                if paramdata in self.stochastic_data:
                    self.stochastic_data_to_objectives_map.\
                        setdefault(paramdata, []).append(objdata)
                    self.objective_to_stochastic_data_map[objdata].\
                        append(paramdata)
            if len(self.objective_to_stochastic_data_map[objdata]) == 0:
                del self.objective_to_stochastic_data_map[objdata]

            obj_variables = \
                tuple(self._collect_unfixed_variables(objdata.expr).values())
            self.objective_to_variables_map[objdata] = []
            for vardata in obj_variables:
                self.variable_to_objectives_map.\
                    setdefault(vardata, []).append(objdata)
                self.objective_to_variables_map[objdata].append(vardata)
            if len(self.objective_to_variables_map[objdata]) == 0:
                del self.objective_to_variables_map[objdata]

        for condata in self.reference_model.component_data_objects(
                Constraint,
                active=True,
                descend_into=True):

            lower_params = \
                tuple(self._collect_mutable_parameters(condata.lower).values())
            body_params = \
                tuple(self._collect_mutable_parameters(condata.body).values())
            upper_params = \
                tuple(self._collect_mutable_parameters(condata.upper).values())

            all_stochastic_params = {}
            for paramdata in itertools.chain(lower_params,
                                             body_params,
                                             upper_params):
                if paramdata in self.stochastic_data:
                    all_stochastic_params[id(paramdata)] = paramdata


            if len(all_stochastic_params) > 0:
                self.constraint_to_stochastic_data_map[condata] = []
                # no params will appear twice in this iteration
                for paramdata in all_stochastic_params.values():
                    self.stochastic_data_to_constraints_map.\
                        setdefault(paramdata, []).append(condata)
                    self.constraint_to_stochastic_data_map[condata].\
                        append(paramdata)

            body_variables = \
                tuple(self._collect_unfixed_variables(condata.body).values())
            self.constraint_to_variables_map[condata] = []
            for vardata in body_variables:
                self.variable_to_constraints_map.\
                    setdefault(vardata, []).append(condata)
                self.constraint_to_variables_map[condata].append(vardata)

        # For now, it is okay to have SOSConstraints in the
        # representation of a problem, but the SOS
        # constraints can't have custom weights that
        # represent stochastic data
        for soscondata in self.reference_model.component_data_objects(
                SOSConstraint,
                active=True,
                descend_into=True):
            for vardata, weight in soscondata.get_items():
                weight_params = tuple(self._collect_mutable_parameters(weight).values())
                if paramdata in self.stochastic_data:
                    raise ValueError("SOSConstraints with stochastic data are currently "
                                     "not supported in implicit stochastic programs. The "
                                     "SOS constraint component '%s' has a weight term for "
                                     "variable '%s' that references stochastic parameter '%s'"
                                     % (soscondata.name,
                                        vardata.name,
                                        paramdata.name))
                self.variable_to_constraints_map.\
                    setdefault(vardata, []).append(condata)
                self.constraint_to_variables_map.\
                    setdefault(condata, []).append(vardata)

        for vardata in self.reference_model.component_data_objects(
                Var,
                descend_into=True):

            lower_params = \
                tuple(self._collect_mutable_parameters(vardata.lb).values())
            self.variable_to_stochastic_data_lb_map[vardata] = []
            for paramdata in lower_params:
                if paramdata in self.stochastic_data:
                    self.stochastic_data_to_variables_lb_map.\
                        setdefault(paramdata, []).append(vardata)
                    self.variable_to_stochastic_data_lb_map[vardata].\
                        append(paramdata)
            if len(self.variable_to_stochastic_data_lb_map[vardata]) == 0:
                del self.variable_to_stochastic_data_lb_map[vardata]

            upper_params = \
                tuple(self._collect_mutable_parameters(vardata.ub).values())
            self.variable_to_stochastic_data_ub_map[vardata] = []
            for paramdata in upper_params:
                if paramdata in self.stochastic_data:
                    self.stochastic_data_to_variables_ub_map.\
                        setdefault(paramdata, []).append(vardata)
                    self.variable_to_stochastic_data_ub_map[vardata].\
                        append(paramdata)
            if len(self.variable_to_stochastic_data_ub_map[vardata]) == 0:
                del self.variable_to_stochastic_data_ub_map[vardata]

    def _extract_stochastic_data(self, model):
        stochastic_data_annotation = locate_annotations(
            model,
            PySP_StochasticDataAnnotation,
            max_allowed=1)
        if len(stochastic_data_annotation) == 0:
            raise ValueError(
                "Reference model is missing stochastic data "
                "annotation: %s" % (PySP_VariableStageAnnotation.__name__))
        else:
            assert len(stochastic_data_annotation) == 1
            stochastic_data_annotation = stochastic_data_annotation[0][1]
        stochastic_data = ComponentMap(
            stochastic_data_annotation.expand_entries())
        if len(stochastic_data) == 0:
            raise ValueError("At least one stochastic data entry is required.")
        for paramdata in stochastic_data:
            assert isinstance(paramdata, _ParamData)
            if paramdata.is_constant():
                raise ValueError(
                    "Stochastic data entry with name '%s' is not mutable. "
                    "All stochastic data parameters must be initialized with "
                    "the mutable keyword set to True." % (paramdata.name))
        return stochastic_data

    def _map_variable_stages(self, model):

        variable_stage_annotation = locate_annotations(
            model,
            PySP_VariableStageAnnotation,
        max_allowed=1)
        if len(variable_stage_annotation) == 0:
            raise ValueError(
                "Reference model is missing variable stage "
                "annotation: %s" % (PySP_VariableStageAnnotation.__name__))
        else:
            assert len(variable_stage_annotation) == 1
            variable_stage_annotation = variable_stage_annotation[0][1]
        variable_stage_assignments = ComponentMap(
            variable_stage_annotation.expand_entries(
                expand_containers=False))
        if len(variable_stage_assignments) == 0:
            raise ValueError("At least one variable stage assignment is required.")

        min_stagenumber = min(variable_stage_assignments.values(),
                              key=lambda x: x[0])[0]
        max_stagenumber = max(variable_stage_assignments.values(),
                              key=lambda x: x[0])[0]
        if max_stagenumber > 2:
            for vardata, (stagenum, derived) in variable_stage_assignments.items():
                if stagenum > 2:
                    raise ValueError(
                        "Implicit stochastic programs must be two-stage, "
                        "but variable with name '%s' has been annotated with "
                        "stage number: %s" % (vardata.name, stagenum))

        stage_to_variables_map = {}
        stage_to_variables_map[1] = []
        stage_to_variables_map[2] = []
        for vardata in model.component_data_objects(
                Var,
                active=True,
                descend_into=True,
                sort=SortComponents.alphabetizeComponentAndIndex):
            stagenumber, derived = variable_stage_assignments.get(vardata, (2, False))
            if (stagenumber != 1) and (stagenumber != 2):
                raise ValueError("Invalid stage annotation for variable with "
                                 "name '%s'. Stage assignment must be 1 or 2. "
                                 "Current value: %s"
                                 % (vardata.name, stagenumber))
            if (stagenumber == 1):
                stage_to_variables_map[1].append((vardata, derived))
            else:
                assert stagenumber == 2
                stage_to_variables_map[2].append((vardata, derived))

        variable_to_stage_map = ComponentMap()
        for stagenum, stagevars in stage_to_variables_map.items():
            for vardata, derived in stagevars:
                variable_to_stage_map[vardata] = (stagenum, derived)

        return (stage_to_variables_map,
                variable_to_stage_map,
                variable_stage_assignments)

    def _generate_base_scenario_tree(self, model, variable_stage_assignments):

        stage_cost_annotation = locate_annotations(
            model,
            PySP_StageCostAnnotation,
            max_allowed=1)
        if len(stage_cost_annotation) == 0:
            raise ValueError("Reference model is missing stage cost "
                             "annotation: %s" % (PySP_StageCostAnnotation.__name__))
        else:
            assert len(stage_cost_annotation) == 1
            stage_cost_annotation = stage_cost_annotation[0][1]
        stage_cost_assignments = ComponentMap(
            stage_cost_annotation.expand_entries())

        stage1_cost = None
        stage2_cost = None
        for cdata, stagenum in stage_cost_assignments.items():
            if stagenum == 1:
                stage1_cost = cdata
            elif stagenum == 2:
                stage2_cost = cdata
        if stage1_cost is None:
            raise ValueError("Missing stage cost annotation for time stage: 1")
        if stage2_cost is None:
            raise ValueError("Missing stage cost annotation for time stage: 2")
        assert stage1_cost != stage2_cost

        #
        # Create a dummy 1-scenario scenario tree
        #

        stm = CreateAbstractScenarioTreeModel()
        stm.Stages.add('Stage1')
        stm.Stages.add('Stage2')
        stm.Nodes.add('RootNode')
        stm.Nodes.add('LeafNode')
        stm.Scenarios.add('ReferenceScenario')
        stm = stm.create_instance()
        stm.NodeStage['RootNode'] = 'Stage1'
        stm.ConditionalProbability['RootNode'] = 1.0
        stm.NodeStage['LeafNode'] = 'Stage2'
        stm.Children['RootNode'].add('LeafNode')
        stm.Children['LeafNode'].clear()
        stm.ConditionalProbability['LeafNode'] = 1.0
        stm.ScenarioLeafNode['ReferenceScenario'] = 'LeafNode'

        stm.StageCost['Stage1'] = stage1_cost.name
        stm.StageCost['Stage2'] = stage2_cost.name
        for var, (stagenum, derived) in variable_stage_assignments.items():
            stagelabel = 'Stage'+str(stagenum)
            if not derived:
                stm.StageVariables[stagelabel].add(var.name)
            else:
                stm.StageDerivedVariables[second_stage].add(var.name)

        scenario_tree = ScenarioTree(scenariotreeinstance=stm)
        scenario_tree.linkInInstances(
            {'ReferenceScenario': self.reference_model})
        return scenario_tree

    @property
    def has_stochastic_objective(self):
        return len(self.objective_to_stochastic_data_map) > 0

    @property
    def has_stochastic_matrix(self):
        return len(self.constraint_to_stochastic_data_body_map) > 0

    @property
    def has_stochastic_rhs(self):
        return (len(self.constraint_to_stochastic_data_lb_map) > 0) or \
            (len(self.constraint_to_stochastic_data_ub_map) > 0)

    @property
    def has_stochastic_variable_bounds(self):
        return (len(self.variable_to_stochastic_data_lb_map) > 0) or \
            (len(self.variable_to_stochastic_data_ub_map) > 0)

    def sample(self):
        raise NotImplementedError

if __name__ == "__main__":

    """
    #
    # test collect_mutable_parameters
    #
    from pyomo.environ import *
    model = ConcreteModel()
    model.p = Param(mutable=True)
    model.q = Param([1], mutable=True, initialize=1.0)
    model.x = Var()

    for obj in [model.p, model.q[1]]:
        assert obj in collect_mutable_parameters(obj).values()
        assert obj in collect_mutable_parameters(obj + 1).values()
        assert obj in collect_mutable_parameters(2 * (obj + 1)).values()
        assert obj in collect_mutable_parameters(2 * obj).values()
        assert obj in collect_mutable_parameters(2 * obj + 1).values()
        assert obj in collect_mutable_parameters(2 * obj + 1 + model.x).values()
        assert obj in collect_mutable_parameters(obj * model.x).values()
        assert obj in collect_mutable_parameters(model.x / obj).values()
        assert obj in collect_mutable_parameters(model.x / (2 * obj)).values()
    """

