#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("EmbeddedSP,")

import os
import itertools
import random
import math

from pyomo.common.collections import ComponentMap
from pyomo.core.expr import current as EXPR
from pyomo.core.base import ComponentUID
from pyomo.core.base.block import (Block,
                                   SortComponents)
from pyomo.core.base.var import Var
from pyomo.core.base.objective import (Objective,
                                       _ObjectiveData)
from pyomo.core.base.constraint import (Constraint,
                                        _ConstraintData)
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.param import _ParamData
from pyomo.pysp.annotations import (locate_annotations,
                                    StageCostAnnotation,
                                    VariableStageAnnotation,
                                    StochasticDataAnnotation,
                                    StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation,
                                    StochasticVariableBoundsAnnotation)
from pyomo.pysp.scenariotree import tree_structure
from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateAbstractScenarioTreeModel
from pyomo.pysp.scenariotree.manager import \
    InvocationType
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManagerClientSerial,
     ScenarioTreeManagerClientPyro)

from six.moves import xrange, zip

# TODO: generate explicit annotations

# TODO: Address the fact that Pyomo variables return
#       numbers when asking for bounds, making it impossible
#       to check if a mutable Param (e.g., stochastic data)
#       appears there.

# generate an absolute path to this file
thisfile = os.path.abspath(__file__)
def _update_data(worker, scenario, data):
    instance = scenario.instance
    assert instance is not None
    for cuid, val in data:
        cuid.find_component_on(instance).value = val

#
# These distributions are documented by the SMPS format
# documentation found at:
# http://myweb.dal.ca/gassmann/smps2.htm#StochIndep
#

class Distribution(object):
    __slots__ = ()
    def expectation(self, *args, **kwds):
        raise NotImplementedError
    def sample(self, *args, **kwds):
        raise NotImplementedError

class TableDistribution(Distribution):
    """
    Table distribution.

    A probability weighted table of discrete values. If no
    weights are provided, the probability of each value is
    considered uniform.
    """
    __slots = ("values", "weights")
    def __init__(self, values, weights=None):
        if len(values) == 0:
            raise ValueError("Empty tables are not allowed")
        self.values = tuple(values)
        if weights is None:
            self.weights = [1.0/len(self.values)]*len(self.values)
        else:
            self.weights = tuple(weights)
        if len(self.values) != len(self.weights):
            raise ValueError("Different number of weights than values")
        if abs(sum(self.weights) - 1) > 1e-6:
            raise ValueError("Weights do not sum to 1")

    def expectation(self):
        return sum(value * weight for value, weight
                   in zip(self.values, self.weights))

    def sample(self):
        x = random.uniform(0, 1)
        cumm = 0.0
        for value, weight in zip(self.values, self.weights):
            cumm += weight
            if x < cumm:
                break
        return value

class UniformDistribution(Distribution):
    """
    Uniform distribution.

    A random number in the range [a, b) or [a, b] depending on rounding.
    """
    __slots__ = ("a", "b")
    def __init__(self, a, b):
        assert a <= b
        self.a = a
        self.b = b

    def expectation(self):
        return (self.b - self.a)/2.0

    def sample(self):
        return random.uniform(self.a, self.b)

class NormalDistribution(Distribution):
    """
    Normal distribution.

    The parameters represent the mean (mu) and the standard
    deviation (sigma).

    The probability density function is:

               e^((x-mu)^2 / -2(sigma^2))
    pdf(x) =  ----------------------------
                sqrt(2 * pi * sigma^2)
    """
    __slots__ = ("mu","sigma","_sampler")
    def __init__(self, mu, sigma, sampler=random.normalvariate):
        self.mu = mu
        self.sigma = sigma
        self._sampler = sampler

    def expectation(self):
        return self.mu

    def sample(self):
        return self._sampler(self.mu, self.sigma)

class LogNormalDistribution(Distribution):
    """
    Log normal distribution.

    A variable x has a log-normal distribution if log(x) is
    normally distributed. The parameters represent the mean
    (mu) and the standard deviation (sigma > 0) of the
    underlying normal distribution.

    The probability density function is:

               e^((ln(x)-mu)^2 / -2(sigma^2))
    pdf(x) =  ----------------------------
                sigma * x * sqrt(2 * pi)
    """
    __slots__ = ("mu","sigma","_sampler")
    def __init__(self, mu, sigma, sampler=random.lognormvariate):
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma
        self._sampler = sampler

    def expectation(self):
        return math.exp(self.mu + ((self.sigma**2)/2.0))

    def sample(self):
        return self._sampler(self.mu, self.sigma)

class GammaDistribution(Distribution):
    """
    Gamma distribution.

    Conditions on the parameters shape > 0 and scale > 0.

    The probability density function is:

               x^(shape-1) * e^(-x/scale)
    pdf(x) =  ----------------------------
              scale^shape * GammaFn(shape)
    """
    __slots__ = ("shape","scale","_sampler")
    def __init__(self, scale, shape, sampler=random.gammavariate):
        assert scale > 0
        assert shape > 0
        self.shape = shape
        self.scale = scale
        self._sampler = sampler

    def expectation(self):
        return self.shape * self.scale

    def sample(self):
        return self._sampler(self.shape, self.scale)

class BetaDistribution(Distribution):
    """
    Beta distribution.

    Conditions on the parameters alpha > 0 and beta > 0.

    The probability density function is:

               x^(alpha-1) * (1-x)^(beta-1)
    pdf(x) =  -----------------------------
                  BetaFn(alpha, beta)
    """
    __slots__ = ("alpha","beta","_sampler")
    def __init__(self, alpha, beta, sampler=random.betavariate):
        assert alpha > 0
        assert alpha > 0
        self.alpha = alpha
        self.beta = beta
        self._sampler = sampler

    def expectation(self):
        return self.alpha / float(self.alpha + self.beta)

    def sample(self):
        return self._sampler(self.alpha, self.beta)

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
        variable_stage_annotation.expand_entries())
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
                    % (var.name, stagenum))

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
                             % (var.name, stagenumber))
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
                % (paramdata.name))
    return stochastic_data

class EmbeddedSP(object):

    @staticmethod
    def _collect_mutable_parameters(exp):
        ans = {}
        for param in EXPR.identify_mutable_parameters(exp):
            ans[id(param)] = param
        return ans

    @staticmethod
    def _collect_variables(exp):
        ans = {}
        for var in EXPR.identify_variables(exp):
            ans[id(var)] = var
        return ans

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

        self.variable_symbols = ComponentMap()

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
        self.variable_symbols = ComponentUID.generate_cuid_string_map(
            self.reference_model, ctype=Var,
            repr_version=tree_structure.CUID_repr_version)
        # remove the parent blocks from this map
        keys_to_delete = []
        for var in self.variable_symbols:
            if var.parent_component().ctype is not Var:
                keys_to_delete.append(var)
        for key in keys_to_delete:
            del self.variable_symbols[key]

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
        assert stage1_cost is not stage2_cost
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
                #       by identifying which variables have
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
            if len(body_params):
                sto_conbody.declare(con)
            if len(body_params) or \
               len(lower_params) or \
               len(upper_params):
                sto_conbounds.declare(con,
                                      lb=bool(len(lower_params) or len(body_params)),
                                      ub=bool(len(upper_params) or len(body_params)))

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
                        % (soscon.name,
                           var.name,
                           param.name))
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
        if sto_obj.has_declarations:
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_objective_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_objective_annotation",
                    sto_obj)
        if sto_conbody.has_declarations:
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_constraint_body_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_constraint_body_annotation",
                    sto_conbody)
        if sto_conbounds.has_declarations:
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_constraint_bounds_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_constraint_bounds_annotation",
                    sto_conbounds)
        if sto_varbounds.has_declarations:
            assert not hasattr(self.reference_model,
                               ".pyspembeddedsp_stochastic_variable_bounds_annotation")
            setattr(self.reference_model,
                    ".pyspembeddedsp_stochastic_variable_bounds_annotation",
                    sto_varbounds)

        # TODO: This is a hack. Cleanup the PySP solver interface
        #       to not require a scenario tree
        #stm = self._create_scenario_tree_model(1)
        #self.scenario_tree = ScenarioTree(scenariotreeinstance=stm)
        #self.scenario_tree.linkInInstances({"s1": self.reference_model})

    def _create_scenario_tree_model(self, size):
        assert size > 0
        stm = CreateAbstractScenarioTreeModel()
        _stages = ["t1", "t2"]
        _nodes = ["root"]
        _scenarios = []
        for i in xrange(1, size+1):
            _nodes.append('n'+str(i))
            _scenarios.append('s'+str(i))
        stm = stm.create_instance(
            data={None: {"Stages": _stages,
                         "Nodes": _nodes,
                         "Scenarios": _scenarios}}
        )
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

        stm.StageCost['t1'] = self._stage1_cost.name
        stm.StageCost['t2'] = self._stage2_cost.name
        for var, (stagenum, derived) in \
              self._variable_stage_assignments.items():
            stage_name = 't'+str(stagenum)
            if not derived:
                stm.StageVariables[stage_name].add(var.name)
            else:
                stm.StageDerivedVariables[stage_name].add(var.name)

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

    def compute_time_stage(self,
                           obj,
                           derived_last_stage=False):
        """
        Determine the time stage that an object belongs
        in. Object types recognized are variables,
        constraints, expressions, and objectives.

        For variables, the time stage is determined by the
        user annotations on the reference model. For
        objectives, constraints, and expressions, the time
        stage is determined by the existance of the
        variables and stochastic data that inside any
        expressions. The time stage is computed as the
        maximum time stage of any variables or stochastic
        data that are encountered. Fixed variables are treated
        as data belonging to the same time stage.

        Args:
            obj: The object to classify.
            derived_last_stage (bool): Indicates whether
                derived variables within a time stage should
                be treated as if they belong to the final
                time stage (where non-anticipativity is not
                enforced). When the value is True, derived
                variables will be treated like variables in
                final time stage. The default is False,
                meaning that the derived status of variables
                will not be considered in the computation.

        Returns:
            The implied time stage for the object. The first
            time stage starts at 1.
        """
        stage = min(self.time_stages)
        laststage = max(self.time_stages)
        # check if the constraint is associated with stochastic data
        # TODO: We need to add the concept of data stage if we
        #       want to deal with the multi-stage case
        if isinstance(obj, _ConstraintData):
            if obj in self.constraint_to_stochastic_data_map:
                stage = laststage
            vars_ = self.constraint_to_variables_map[obj]
        elif isinstance(obj, _ObjectiveData):
            if obj in self.objective_to_stochastic_data_map:
                stage = laststage
            vars_ = self.objective_to_variables_map[obj]
        else:
            vars_ = tuple(self._collect_variables(obj).values())
        # there is no point in executing this check if we
        # already know that the object belongs to the final
        # time stage
        if stage < laststage:
            for var in vars_:
                varstage, derived = self.variable_to_stage_map[var]
                if derived_last_stage and derived:
                    stage = laststage
                else:
                    stage = max(stage, varstage)
                if stage == laststage:
                    # no point in checking anything else
                    break
            else: # executed when no break occurs in the for loop
                stage = 1
        return stage

    def pyro_sample_sp(self,
                       size,
                       **kwds):
        assert size > 0
        model = self.reference_model.clone()

        scenario_tree_model = \
            self._create_scenario_tree_model(size)
        factory = ScenarioTreeInstanceFactory(
            model=self.reference_model,
            scenario_tree=scenario_tree_model)
        options = \
            ScenarioTreeManagerClientPyro.register_options()
        for key in kwds:
            options[key] = kwds[key]
        manager = ScenarioTreeManagerClientPyro(
            options,
            factory=factory)
        try:
            init = manager.initialize(async_call=True)
            pcuids = ComponentMap()
            for param in self.stochastic_data:
                pcuids[param] = ComponentUID(param)
            init.complete()
            for scenario in manager.scenario_tree.scenarios:
                data = []
                for param, dist in self.stochastic_data.items():
                    data.append((pcuids[param], dist.sample()))
                manager.invoke_function(
                    "_update_data",
                    thisfile,
                    invocation_type=InvocationType.OnScenario(scenario.name),
                    function_args=(data,),
                    oneway_call=True)
            manager.reference_model = model
        except:
            manager.close()
            raise
        return manager

    def generate_sample_sp(self, size, **kwds):
        assert size > 0
        def model_callback(scenario_name, node_list):
            m = self.sample(return_copy=True)
            return m
        scenario_tree_model = self._create_scenario_tree_model(size)
        factory = ScenarioTreeInstanceFactory(
            model=model_callback,
            scenario_tree=scenario_tree_model)
        options = \
            ScenarioTreeManagerClientSerial.register_options()
        for key in kwds:
            options[key] = kwds[key]
        manager = ScenarioTreeManagerClientSerial(options,
                                                  factory=factory)
        manager.initialize()
        manager.reference_model = self.reference_model.clone()
        return manager

    def sample(self, return_copy=False):
        for param, dist in self.stochastic_data.items():
            param.value = dist.sample()
        if return_copy:
            return self.reference_model.clone()

    def set_expected_value(self, return_copy=False):
        for param, dist in self.stochastic_data.items():
            param.value = dist.expectation()
        if return_copy:
            return self.reference_model.clone()
