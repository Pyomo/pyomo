#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("EmbeddedSP,")

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
                                    StageCostAnnotation,
                                    VariableStageAnnotation,
                                    StochasticDataAnnotation,
                                    StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation,
                                    StochasticVariableBoundsAnnotation)
from pyomo.pysp.scenariotree.tree_structure import ScenarioTree
from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateAbstractScenarioTreeModel
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.manager_solver import \
    ScenarioTreeManagerSolverClientSerial

from six.moves import xrange, zip

# TODO: generate explicit annotations

# TODO: Address the fact that Pyomo variables return
#       numbers when asking for bounds, making it impossible
#       to check if a mutable Param (e.g., stochastic data)
#       appears there.

"""
These distributions are documented by the SMPS format
documentation found at: http://myweb.dal.ca/gassmann/smps2.htm#StochIndep
"""

import random
import math
class NormalDistribution(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def sample(self):
        return random.normalvariate(self.mu, self.sigma)

"""
class PySPDistribution(object):
    def sample(self, *args, **kwds):
        raise NotImplementedError

class TableDistribution(PySPDistribution):
    def __init__(self, values, weights=None):
        if weights is None:
            weights = [1.0/len(values)]*len(values)
        else:
            assert len(weights) == len(values)
            assert abs(sum(weights) - 1) < 1e-6
        super(DiscreteDistribution, self).__init__(values, weights, "DISCRETE")

class NormalDistribution(PySPDistribution):
    def __init__(self, mean, variance):
        assert variance >= 0
        super(NormalDistribution, self).__init__(mean, variance, "NORMAL")

class UniformDistribution(PySPDistribution):
    def __init__(self, a, b):
        assert a <= b
        super(UniformDistribution, self).__init__(a, b, "UNIFORM")

class GammaDistribution(PySPDistribution):
    def __init__(self, scale, shape):
        assert scale >= 0
        assert shape >= 0
        super(GammaDistribution, self).__init__(scale, shape, "GAMMA")

class BetaDistribution(PySPDistribution):
    def __init__(self, alpha, beta):
        assert alpha > 0
        assert beta > 0
        super(BetaDistribution, self).__init__(alpha, beta, "BETA")

class LogNormalDistribution(PySPDistribution):
    def __init__(self, mean, variance):
        assert variance >= 0
        super(LogNormalDistribution, self).__init__(mean, variance, "LOGNORM")
"""

def _map_variable_stages(model):

    variable_stage_annotation = locate_annotations(
        model,
        VariableStageAnnotation,
    max_allowed=1)
    if len(variable_stage_annotation) == 0:
        raise ValueError(
            "Reference model is missing variable stage "
            "annotation: %s" % (VariableStageAnnotation.__name__))
    else:
        assert len(variable_stage_annotation) == 1
        variable_stage_annotation = variable_stage_annotation[0][1]
    variable_stage_assignments = ComponentMap(
        variable_stage_annotation.expand_entries(
            expand_containers=False))
    if len(variable_stage_assignments) == 0:
        raise ValueError("At least one variable stage assignment "
                         "is required.")

    min_stagenumber = min(variable_stage_assignments.values(),
                          key=lambda x: x[0])[0]
    max_stagenumber = max(variable_stage_assignments.values(),
                          key=lambda x: x[0])[0]
    if max_stagenumber > 2:
        for var, (stagenum, derived) in \
              variable_stage_assignments.items():
            if stagenum > 2:
                raise ValueError(
                    "Embedded stochastic programs must be two-stage "
                    "(for now), but variable with name '%s' has been "
                    "annotated with stage number: %s"
                    % (var.cname(True), stagenum))

    stage_to_variables_map = {}
    stage_to_variables_map[1] = []
    stage_to_variables_map[2] = []
    for var in model.component_data_objects(
            Var,
            active=True,
            descend_into=True,
            sort=SortComponents.alphabetizeComponentAndIndex):
        stagenumber, derived = \
            variable_stage_assignments.get(var, (2, False))
        if (stagenumber != 1) and (stagenumber != 2):
            raise ValueError("Invalid stage annotation for variable with "
                             "name '%s'. Stage assignment must be 1 or 2. "
                             "Current value: %s"
                             % (var.cname(True), stagenumber))
        if (stagenumber == 1):
            stage_to_variables_map[1].append((var, derived))
        else:
            assert stagenumber == 2
            stage_to_variables_map[2].append((var, derived))

    variable_to_stage_map = ComponentMap()
    for stagenum, stagevars in stage_to_variables_map.items():
        for var, derived in stagevars:
            variable_to_stage_map[var] = (stagenum, derived)

    return (stage_to_variables_map,
            variable_to_stage_map,
            variable_stage_assignments)

def _extract_stochastic_data(model):
    stochastic_data_annotation = locate_annotations(
        model,
        StochasticDataAnnotation,
        max_allowed=1)
    if len(stochastic_data_annotation) == 0:
        raise ValueError(
            "Reference model is missing stochastic data "
            "annotation: %s" % (StochasticDataAnnotation.__name__))
    else:
        assert len(stochastic_data_annotation) == 1
        stochastic_data_annotation = stochastic_data_annotation[0][1]
    stochastic_data = ComponentMap(
        stochastic_data_annotation.expand_entries())
    if len(stochastic_data) == 0:
        raise ValueError("At least one stochastic data "
                         "entry is required.")
    for paramdata in stochastic_data:
        assert isinstance(paramdata, _ParamData)
        if paramdata.is_constant():
            raise ValueError(
                "Stochastic data entry with name '%s' is not mutable. "
                "All stochastic data parameters must be initialized "
                "with the mutable keyword set to True."
                % (paramdata.cname(True)))
    return stochastic_data

class EmbeddedSP(object):

    @staticmethod
    def _collect_mutable_parameters(exp):
        """
        A helper function for querying a pyomo expression for
        mutable parameters.
        """
        if is_constant(exp) or isinstance(exp, _VarData):
            return {}
        elif exp.is_expression():
            ans = {}
            if exp.__class__ is pyomo.core.base.expr._ProductExpression:
                for subexp in exp._numerator:
                    ans.update(EmbeddedSP.\
                               _collect_mutable_parameters(subexp))
                for subexp in exp._denominator:
                    ans.update(EmbeddedSP.\
                               _collect_mutable_parameters(subexp))
            else:
                # This is fragile: we assume that all other expression
                # objects "play nice" and just use the _args member.
                for subexp in exp._args:
                    ans.update(EmbeddedSP.\
                               _collect_mutable_parameters(subexp))
            return ans
        elif isinstance(exp, _ParamData):
            return {id(exp): exp}
        else:
            raise ValueError("Unexpected expression type: "+str(exp))

    @staticmethod
    def _collect_variables(exp):
        if is_constant(exp):
            return {}
        elif exp.is_expression():
            ans = {}
            if exp.__class__ is pyomo.core.base.expr._ProductExpression:
                for subexp in exp._numerator:
                    ans.update(EmbeddedSP.\
                               _collect_variables(subexp))
                for subexp in exp._denominator:
                    ans.update(EmbeddedSP.\
                               _collect_variables(subexp))
            else:
                # This is fragile: we assume that all other expression
                # objects "play nice" and just use the _args member.
                for subexp in exp._args:
                    ans.update(EmbeddedSP.\
                               _collect_variables(subexp))
            return ans
        elif isinstance(exp, _VarData):
            return {id(exp): exp}
        elif is_fixed(exp):
            return {}
        else:
            raise ValueError("Unexpected expression type: "+str(exp))

    def __init__(self, reference_model):

        self.reference_model = None
        self.objective = None
        self.time_stages = None

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

        # maps between stochastic data and variable lower and upper bounds
        self.stochastic_data_to_variables_lb_map = ComponentMap()
        self.variable_to_stochastic_data_lb_map = ComponentMap()

        self.stochastic_data_to_variables_ub_map = ComponentMap()
        self.variable_to_stochastic_data_ub_map = ComponentMap()

        if not isinstance(reference_model, Block):
            raise TypeError("reference model input must be a Pyomo model")
        self.reference_model = reference_model

        #
        # Extract stochastic parameters from the
        # StochasticDataAnnotation object
        #
        self.stochastic_data = \
            _extract_stochastic_data(self.reference_model)

        #
        # Get the variable stages from the
        # VariableStageAnnotation object
        #
        (self.stage_to_variables_map,
         self.variable_to_stage_map,
         self._variable_stage_assignments) = \
            _map_variable_stages(self.reference_model)
        self.time_stages = tuple(sorted(self.stage_to_variables_map))
        assert self.time_stages[0] == 1

        #
        # Get the stage cost components from the StageCostAnnotation
        # and generate a dummy single-scenario scenario tree
        #
        stage_cost_annotation = locate_annotations(
            self.reference_model,
            StageCostAnnotation,
            max_allowed=1)
        if len(stage_cost_annotation) == 0:
            raise ValueError(
                "Reference model is missing stage cost "
                "annotation: %s" % (StageCostAnnotation.__name__))
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
            raise ValueError("Missing stage cost annotation "
                             "for time stage: 1")
        if stage2_cost is None:
            raise ValueError("Missing stage cost annotation "
                             "for time stage: 2")
        assert stage1_cost != stage2_cost
        self._stage1_cost = stage1_cost
        self._stage2_cost = stage2_cost

        #
        # Extract the locations of variables and stochastic data
        # within the model
        #
        sto_obj = StochasticObjectiveAnnotation()
        for objcntr, obj in enumerate(
                  self.reference_model.component_data_objects(
                Objective,
                active=True,
                descend_into=True), 1):

            if objcntr > 1:
                raise ValueError(
                    "Reference model can not contain more than one "
                    "active objective")

            self.objective = obj
            self.objective_sense = obj.sense

            obj_params = tuple(
                self._collect_mutable_parameters(obj.expr).values())
            self.objective_to_stochastic_data_map[obj] = []
            for paramdata in obj_params:
                if paramdata in self.stochastic_data:
                    self.stochastic_data_to_objectives_map.\
                        setdefault(paramdata, []).append(obj)
                    self.objective_to_stochastic_data_map[obj].\
                        append(paramdata)
            if len(self.objective_to_stochastic_data_map[obj]) == 0:
                del self.objective_to_stochastic_data_map[obj]
            else:
                # TODO: Can we make this declaration sparse
                #       by idenfifying which variables have
                #       stochastic coefficients? How to handle
                #       non-linear expressions?
                sto_obj.declare(obj)

            obj_variables = tuple(
                self._collect_variables(obj.expr).values())
            self.objective_to_variables_map[obj] = []
            for var in obj_variables:
                self.variable_to_objectives_map.\
                    setdefault(var, []).append(obj)
                self.objective_to_variables_map[obj].append(var)
            if len(self.objective_to_variables_map[obj]) == 0:
                del self.objective_to_variables_map[obj]

        sto_conbounds = StochasticConstraintBoundsAnnotation()
        sto_conbody = StochasticConstraintBodyAnnotation()
        for con in self.reference_model.component_data_objects(
                Constraint,
                active=True,
                descend_into=True):

            lower_params = tuple(
                self._collect_mutable_parameters(con.lower).values())
            body_params = tuple(
                self._collect_mutable_parameters(con.body).values())
            upper_params = tuple(
                self._collect_mutable_parameters(con.upper).values())

            # TODO: Can we make this declaration sparse
            #       by idenfifying which variables have
            #       stochastic coefficients? How to handle
            #       non-linear expressions? Currently, this
            #       code also fails to detect that mutable
            #       "constant" expressions might fall out
            #       of the body and into the bounds.
            if len(body_params) > 0:
                sto_conbody.declare(con)
            if (len(lower_params) > 0) or \
               (len(upper_params) > 0):
                sto_conbounds.declare(con,
                                      lb=bool(len(lower_params) > 0),
                                      ub=bool(len(upper_params) > 0))

            all_stochastic_params = {}
            for param in itertools.chain(lower_params,
                                         body_params,
                                         upper_params):
                if param in self.stochastic_data:
                    all_stochastic_params[id(param)] = param

            if len(all_stochastic_params) > 0:
                self.constraint_to_stochastic_data_map[con] = []
                # no params will appear twice in this iteration
                for param in all_stochastic_params.values():
                    self.stochastic_data_to_constraints_map.\
                        setdefault(param, []).append(con)
                    self.constraint_to_stochastic_data_map[con].\
                        append(param)

            body_variables = tuple(
                self._collect_variables(con.body).values())
            self.constraint_to_variables_map[con] = []
            for var in body_variables:
                self.variable_to_constraints_map.\
                    setdefault(var, []).append(con)
                self.constraint_to_variables_map[con].append(var)

        # For now, it is okay to have SOSConstraints in the
        # representation of a problem, but the SOS
        # constraints can't have custom weights that
        # represent stochastic data
        for soscon in self.reference_model.component_data_objects(
                SOSConstraint,
                active=True,
                descend_into=True):
            for var, weight in soscon.get_items():
                weight_params = tuple(
                    self._collect_mutable_parameters(weight).values())
                if param in self.stochastic_data:
                    raise ValueError(
                        "SOSConstraints with stochastic data are currently"
                        " not supported in embedded stochastic programs. "
                        "The SOSConstraint component '%s' has a weight "
                        "term for variable '%s' that references stochastic"
                        " parameter '%s'"
                        % (soscon.cname(True),
                           var.cname(True),
                           param.cname(True)))
                self.variable_to_constraints_map.\
                    setdefault(var, []).append(soscon)
                self.constraint_to_variables_map.\
                    setdefault(soscon, []).append(var)

        sto_varbounds = StochasticVariableBoundsAnnotation()
        for var in self.reference_model.component_data_objects(
                Var,
                descend_into=True):

            lower_params = tuple(
                self._collect_mutable_parameters(var.lb).values())
            upper_params = tuple(
                self._collect_mutable_parameters(var.ub).values())

            if (len(lower_params) > 0) or \
               (len(upper_params) > 0):
                sto_varbounds.declare(var,
                                      lb=bool(len(lower_params) > 0),
                                      ub=bool(len(upper_params) > 0))

            self.variable_to_stochastic_data_lb_map[var] = []
            for param in lower_params:
                if param in self.stochastic_data:
                    self.stochastic_data_to_variables_lb_map.\
                        setdefault(param, []).append(var)
                    self.variable_to_stochastic_data_lb_map[var].\
                        append(param)
            if len(self.variable_to_stochastic_data_lb_map[var]) == 0:
                del self.variable_to_stochastic_data_lb_map[var]

            self.variable_to_stochastic_data_ub_map[var] = []
            for param in upper_params:
                if param in self.stochastic_data:
                    self.stochastic_data_to_variables_ub_map.\
                        setdefault(param, []).append(var)
                    self.variable_to_stochastic_data_ub_map[var].\
                        append(param)
            if len(self.variable_to_stochastic_data_ub_map[var]) == 0:
                del self.variable_to_stochastic_data_ub_map[var]

        #
        # Generate the explicit annotations
        #

        # first make sure these annotations do not already exist
        if len(locate_annotations(self.reference_model,
                                  StochasticConstraintBoundsAnnotation)) > 0:
            raise ValueError("Reference model can not contain "
                             "a StochasticConstraintBoundsAnnotation declaration.")
        if len(locate_annotations(self.reference_model,
                                  StochasticConstraintBodyAnnotation)) > 0:
            raise ValueError("Reference model can not contain "
                             "a StochasticConstraintBodyAnnotation declaration.")
        if len(locate_annotations(self.reference_model,
                                  StochasticObjectiveAnnotation)) > 0:
            raise ValueError("Reference model can not contain "
                             "a StochasticObjectiveAnnotation declaration.")

        # now add any necessary annotations
        if sto_obj.has_declarations():
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_objective_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_objective_annotation",
                    sto_obj)
        if sto_conbody.has_declarations():
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_constraint_body_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_constraint_body_annotation",
                    sto_conbody)
        if sto_conbounds.has_declarations():
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_constraint_bounds_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_constraint_bounds_annotation",
                    sto_conbounds)
        if sto_varbounds.has_declarations():
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_variable_bounds_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_variable_bounds_annotation",
                    sto_varbounds)

        # TODO: This is a hack. Cleanup the PySP solver interface
        #       to not require a scenario tree
        stm = self._create_scenario_tree_model(1)
        self.scenario_tree = ScenarioTree(scenariotreeinstance=stm)
        self.scenario_tree.linkInInstances({"s1": self.reference_model})

    def _create_scenario_tree_model(self, size):
        assert size > 0
        stm = CreateAbstractScenarioTreeModel()
        stm.Stages.add('t1')
        stm.Stages.add('t2')
        stm.Nodes.add('root')
        for i in xrange(1, size+1):
            stm.Nodes.add('n'+str(i))
            stm.Scenarios.add('s'+str(i))
        stm = stm.create_instance()
        stm.NodeStage['root'] = 't1'
        stm.ConditionalProbability['root'] = 1.0
        weight = 1.0/float(size)
        for i in xrange(1, size+1):
            node_name = 'n'+str(i)
            scen_name = 's'+str(i)
            stm.NodeStage[node_name] = 't2'
            stm.Children['root'].add(node_name)
            stm.Children[node_name].clear()
            stm.ConditionalProbability[node_name] = weight
            stm.ScenarioLeafNode[scen_name] = node_name

        stm.StageCost['t1'] = self._stage1_cost.cname(True)
        stm.StageCost['t2'] = self._stage2_cost.cname(True)
        for var, (stagenum, derived) in \
              self._variable_stage_assignments.items():
            stage_name = 't'+str(stagenum)
            if not derived:
                stm.StageVariables[stage_name].add(var.cname(True))
            else:
                stm.StageDerivedVariables[stage_name].add(var.cname(True))

        return stm

    @property
    def has_stochastic_objective(self):
        """Returns whether the SP has a stochastic data in the objective."""
        return len(self.objective_to_stochastic_data_map) > 0

    @property
    def has_stochastic_constraints(self):
        """Returns whether the SP has stochastic data in the body of any constraints."""
        return len(self.constraint_to_stochastic_data_map) > 0

    @property
    def has_stochastic_variable_bounds(self):
        """Returns whether the SP has stochastic data in the bounds of any variables."""
        return (len(self.variable_to_stochastic_data_lb_map) > 0) or \
            (len(self.variable_to_stochastic_data_ub_map) > 0)

    def compute_constraint_stage(self,
                                 constraint_object,
                                 derived_last_stage=False,
                                 check_fixed_status=True):
        """
        Obtain the time stage that a constraint belongs in.

        Computes the time stage of a constraint based on
        the time stage of the variables and stochastic data that appear
        in the constraint's lower bound, body, and upper
        bound expressions. The time stage for the constraint is
        computed as the maximum time stage of any variables or
        stochastic data that are encountered.

        Args:
            constraint_object: The constraint to inspect.
            derived_last_stage (bool): Indicates that
                derived variables within a time stage should
                be treated as if they belong to the final
                time stage when computing the time stage of
                the constraint. When the value is True,
                derived variables will be treated like
                variables in final time stage. The default
                is False, meaning that the derived status of
                variables will not be considered in the
                computation.
            check_fixed_status (bool): Indicates that the
                fixed status of variables should be
                considered when computing the time stage of
                the constraint. When the value is False, the
                fixed status of variables will be ignored.
                The default is True, meaning that variables
                whose fixed flag is active will be treated
                as constants and excluded from the
                computation.

        Returns:
            The implied time stage for the constraint.
        """
        stage = min(self.time_stages)
        laststage = max(self.time_stages)
        # check if the constraint is associated with stochastic data
        # TODO: We need to add the concept of data stage if we
        #       want to deal with the multi-stage case
        if constraint_object in self.constraint_to_stochastic_data_map:
            stage = laststage
        # there is no point in executing this check if we
        # already know the constraint belongs to the final
        # time stage
        if stage < laststage:
            for var in \
                  self.constraint_to_variables_map[constraint_object]:
                if (not var.fixed) or (not check_fixed_status):
                    varstage, derived = self.variable_to_stage_map[var]
                    if derived_last_stage and derived:
                        stage = laststage
                    else:
                        stage = max(stage, varstage)
                    if stage == laststage:
                        # no point in checking anything else
                        break
        return stage

    def generate_sample_sp(self, size, options=None):
        assert size > 0
        def model_callback(scenario_name, node_list):
            m = self.sample(return_copy=True)
            # TODO
            del m._PySP_UserCostExpression
            return m
        scenario_tree_model = self._create_scenario_tree_model(size)
        factory = ScenarioTreeInstanceFactory(model_callback,
                                              scenario_tree_model)
        if options is None:
            options = \
                ScenarioTreeManagerSolverClientSerial.register_options()
        manager = ScenarioTreeManagerSolverClientSerial(options,
                                                        factory=factory)
        manager.initialize()
        return manager

    def sample(self, return_copy=False):
        for param, dist in self.stochastic_data.items():
            param.value = dist.sample()
        if return_copy:
            return self.reference_model.clone()
