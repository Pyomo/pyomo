#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import numpy as np

from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Block, Var, Param, VarList, ConstraintList, Constraint, Objective,
    RangeSet, value, ObjectiveList
)
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.contrib.trustregion.geometry import generate_geometry, quadraticExpression
from pyomo.contrib.trustregion.utils import maxIgnoreNone, minIgnoreNone

logger = logging.getLogger('pyomo.contrib.trustregion')

class RMType:
    linear = 0
    quadratic = 1


class ReplaceEFVisitor(EXPR.ExpressionReplacementVisitor):
    # I am making the assumption that after we rework this,
    # the data structures will be better.
    # Specifically:
    #    TRF.ef_variables : Dict (for which the keys are TRF.external_fcns)
    #    { 'externalNode': {'index': #, 'vars': [vars]}}
    # There is no need for TRF.external_fcns given this new structure
    pass


class PyomoInterface(object):
    """
    Initialize with a pyomo model m.
    This is used in TrustRegionMethod.py, same requirements for m apply

    m is reformulated into form for use in TRF algorithm

    Specified ExternalFunction() objects are replaced with new variables
    All new attributes (including these variables) are stored on block
    "tR"

    """
    rmtype = RMType.linear

    def __init__(self, m, efList, config):

        self.config = config
        self.model = m
        self.TRF = self.transformForTrustRegion(self.model, efList)

        self.numberOfInputs = len(self.TRF.ef_inputs)
        self.numberOfOutputs = len(self.TRF.ef_outputs)
        self.numberOfOtherVars = len(self.TRF.other_vars)

        self.createParam()
        self.createRMConstraint()
        self.cacheBound()

        self.optimalMatrix = None
        self.optimalPointSet = None

    def __exit__(self):

        self.reactivateObjects(self.model)

    def substituteEF(self, expr, TRF, efSet):
        """
        Substitute out an External Function

        Arguments:
            expr : a pyomo expression. We will search this expression tree
            TRF : a pyomo block. We will add tear variables y on this block
            efSet: the (pyomo) set of external functions for which we will
                   use TRF method

        This function returns an expression after removing any
        ExternalFunction in the set efSet from the expression tree
        expr. New variables are declared on the trf block and replace
        the external function.

        """
        return ReplaceEFVisitor(TRF, efSet).dfs_postorder_stack(expr)

    def transformForTrustRegion(self, model, efList):
        """
        Convert model into a suitable format for TRF method
        
        Deactivates objects that have been replaced by TRF forms
        
        Arguments:
            model  : pyomo model containing ExternalFunctions
            efList : a list of external functions that will be handled
                     with the TRF method, rather than calls to compiled code
        """
        efSet = set([id(ef) for ef in efList])
        TRF = Block()

        allVariables = set([var for var in model.component_data_objects(Var)])

        model.add_component(unique_component_name(model, 'tR'), TRF)
        TRF.ef_outputs = VarList()
        TRF.constraints_ref_ef = ConstraintList()
        TRF.objective_ref_ef = ObjectiveList()
        TRF.ef_variables = {}
        TRF.deactivated_objects = []

        # Transform and add Constraints to TRF model
        for con in model.component_data_objects(Constraint, active=True):
            origbody = con.body
            newbody = self.substituteEF(origbody, TRF, efSet)
            if origbody is not newbody:
                TRF.constraints_ref_ef.add((con.lower, newbody, con.upper))
                con.deactivate()
                TRF.deactivated_objects.append(con)

        # Transform and add Objective to TRF model
        objs = list(model.component_data_objects(Objective, active=True))
        if len(objs) != 1:
            raise RuntimeError(
                """transformForTrustRegion:
  TrustRegion only supports models with a single active Objective.""")
        origobj = objs[0].expr
        newobj = self.substituteEF(origobj, TRF, efSet)
        if origobj is not newobj:
            TRF.objective_ref_ef.add(newobj)
            objs[0].deactivate()
            TRF.deactivated_objects.append(objs[0])

        TRF.ef_inputs = []
        seen = ComponentMap()
        self.ef_input_idxs = ComponentMap()
        for externalNode in TRF.ef_variables.keys():
            self.ef_input_idxs[externalNode] = []
            # NOTE: identify_variables is guaranteed to not return duplicates
            for var in EXPR.identify_variables(externalNode, include_fixed=False):
                if var not in seen:
                    seen[v] = len(TRF.ef_inputs)
                    TRF.ef_inputs.append(var)
                self.ef_input_idxs[externalNode].append(seen[var])

        TRF.other_vars = []
        # Remaining variables from allVariables are added to other_vars
        for var in allVariables:
            if id(var) not in seen:
                seen.add(id(var))
                TRF.other_vars.append(var)

        return TRF
            
    def reactivateObjects(self, model):
        """
        Reactivate objects that were deactivated as a result
        of the transformForTrustRegion method
        """
        for obj in model.TRF.deactivated_objects:
            obj.activate()

    def getInitialValue(self):
        """
        Initialize values into numpy array
        """
        # We initialize outputs as 1s to start
        outputs = np.ones(self.numberOfOutputs, dtype=float)
        inputs = np.zeros(self.numberOfInputs, dtype=float)
        other_vars = np.zeros(self.numberOfOtherVars, dtype=float)
        for i in range(self.numberOfInputs):
            inputs[i] = value(self.TRF.ef_inputs[i])
        for i in range(self.numberOfOtherVars):
            other_vars[i] = value(self.TRF.other_vars[i])
        return inputs, outputs, other_vars

    def createParam(self):
        """
        Create relevant parameters
        """
        ind_input = RangeSet(0, self.numberOfInputs-1)
        self.TRF.ind_output = RangeSet(0, self.numberOfOutputs-1)
        ind_other = RangeSet(0, self.numberOfOtherVars-1)
        self.TRF.initInput = Param(ind_input,
                            mutable=True, default=0)
        self.TRF.initOutput = Param(self.TRF.ind_output,
                            mutable=True, default=0)
        self.TRF.initOther = Param(ind_other,
                            mutable=True, default=0)
        if self.rmtype == RMType.linear:
            self.TRF.rmParams = Param(self.TRF.ind_output,
                                     range(self.numberOfInputs + 1),
                                     mutable=True, default=0)
        elif self.rmtype == RMType.quadratic:
            self.TRF.rmParams = Param(self.TRF.ind_output,
                                     range(quadraticExpression(self.numberOfInputs)),
                                     mutable=True, default=0)
        else:
            raise RuntimeError(
                "createParam: TrustRegion only supports linear and quadratic RM types.")

    def RMConstraint(self, model, externalNode):
        """
        Expression for use in the RM Constraint
        """
        nodeIndex = self.TRF.ef_variables[externalNode]['index']
        constantTerm = model.rmParams[externalNode][idx, 0]
        linearTerms = model.rmParams[externalNode][idx, 1:self.numberOfInputs+1]
        quad = model.rmParams[externalNode][idx, self.numberOfInputs+1:]
        quadraticTerms = { (i, j) : \
                    quad[i*(2*self.numberOfInputs - 1 - i)//2 + j for i in\
                    range(self.numberOfInputs) for j in range(i, self.numberOfInputs)] }
        expr = (
            constantTerm
            + sum(linearTerms[i] * (model.ef_inputs[i] - model.init_inputs[i]) for i in range(self.numberOfInputs))
            )
        if (self.rmtype == RMType.quadratic):
            quadraticTerms = model.rmParams[externalNode]['quadratic']
            expr += sum(quadraticTerms[(i, j)] * (model.ef_inputs[i] \
                    - model.init_inputs[i]) * (model.ef_inputs[j] - model.init_inputs[j])\
                    for i in range(self.numberOfInputs) \
                    for j in range(self.numberOfInputs))
        return expr

    def createRMConstraint(self):
        """
        Create appropriate constraints for RM
        """
        def RMConstraintRule(model, externalNode):
            idx = self.TRF.ef_variables[externalNode]['index']
            return model.y[idx] == self.RMConstraint(model, externalNode)
        self.TRF.RMConstraint = Constraint(self.TRF.ind_ly,
                                           rule=RMConstraintRule)

    def cacheBound(self):
        """
        Store the upper and lower bounds for each variable
        """
        self.TRF.ef_input_lower = []
        self.TRF.ef_input_upper = []
        self.TRF.other_lower = []
        self.TRF.other_upper = []
        for i in self.TRF.ef_inputs:
            self.TRF.ef_input_lower.append(i.lb)
            self.TRF.ef_input_upper.append(i.up)
        for ov in self.TRF.other_vars:
            self.TRF.other_lower.append(ov.lb)
            self.TRF.other_upper.append(ov.up)

    def setParam(self, inputs=None, RMParams=None):
        """
        Populate parameter values
        """
        if inputs is not None:
            self.TRF.init_inputs.store_values(inputs)
        if RMParams is not None:
            externalNodes = self.TRF.ef_variables.keys()
            for externalNode in externalNodes:
                idx = self.TRF.ef_variables[externalNode]['index']
                for j in range(len(RMParams[externalNode])):
                    self.TRF.rmParams[idx, j] = RMParams[externalNode][j]

    def setVarValue(self, inputs=None, outputs=None, other=None):
        """
        Populate variable values
        """
        if inputs is not None:
            if(len(inputs) != self.numberOfInputs):
                raise Exception(
                    "setVarValue: The dimension of inputs is not consistent.\n")
            self.TRF.ef_inputs.set_values(inputs)
        if outputs is not None:
            if(len(outputs) != self.numberOfOutputs):
                raise Exception(
                    "setVarValue: The dimension of outputs is not consistent.\n")
            self.TRF.ef_outputs.set_values(outputs)
        if other is not None:
            if(len(other) != self.numberOfOtherVars):
                raise Exception(
                    "setVarValue: The dimension of glass-box variables is not consistent.\n")
            self.TRF.other_vars.set_values(other)

    def setBound(self, inputs, outputs, other, radius):
        """
        Set bounds for variables
        """
        for i in range(self.numberOfInputs):
            self.TRF.ef_inputs[i].setlb(maxIgnoreNone(inputs[i] -
                                                  radius, self.TRF.ef_input_lower[i]))
            self.TRF.ef_inputs[i].setub(minIgnoreNone(inputs[i] +
                                                  radius, self.TRF.ef_input_upper[i]))
        for i in range(self.numberOfOutputs):
            self.TRF.ef_outputs[i].setlb(outputs[i] - radius)
            self.TRF.ef_outputs[i].setub(outputs[i] + radius)
        for i in range(self.numberOfOtherVars):
            self.TRF.other_vars[i].setlb(maxIgnoreNone(other[i] -
                                                  radius, self.TRF.other_lower[i]))
            self.TRF.other_vars[i].setub(minIgnoreNone(other[i] +
                                                  radius, self.TRF.other_upper[i]))

    def getEFInputValues(self):
        """
        Cache current EF input values
        """
        return curr_vals = list(var.value for var in self.TRF.ef_inputs)

    def setEFInputValues(self, inputs):
        """
        Set new values for EF inputs
        """
        for var, val in zip(self.TRF.ef_inputs, inputs):
            var.set_value(val)

    def evaluateEF(self, inputs):
        """
        Evaluate values at external functions
        """
        externalNodes = self.TRF.ef_variables.keys()
        self.setEFInputValues(inputs)
        outputs = list(value(externalNode) for externalNode in externalNodes)
        return np.array(outputs)

    def evaluateObj(self, inputs, outputs, other):
        """
        Evaluate the objective
        """
        self.setVarValue(inputs=inputs, outputs=outputs, other=other)
        return self.objective()

    def solveModel(self):
        """
        Method to trigger solver
        """
        model = self.model
        opt = SolverFactory(self.config.solver)
        opt.options.update(self.config.solver_options)

        results = opt.solve(
            model, keepfiles=self.config.keepfiles, tee=self.config.tee)

        if ((results.solver.status == SolverStatus.ok)
                and (results.solver.termination_condition == TerminationCondition.optimal)):
            model.solutions.load_from(results)

            for obj in model.component_data_objects(Objective, active=True):
                return True, obj()
        else:
            print("Warning: Solver Status: " + str(results.solver.status))
            print("Termination Condition(s): " + str(results.solver.termination_condition))
            return False, 0

    def TRSPk(self, inputs, outputs, other,
              init_inputs, init_outputs, init_other,
              RMParams, radius):
        """
        Trust region subproblem solver

        Used for each iteration of the trust region filter method
        """
        if ((len(inputs) != self.numberOfInputs)
            or (len(outputs) != self.numberOfOutputs)
            or (len(other) != self.numberOfOtherVars)
            or (len(init_inputs) != self.numberOfInputs)
            or (len(init_outputs) != self.numberOfOutputs)
            or (len(init_other) != self.numberOfOtherVars)):
            raise RuntimeError(
                "TRSP_k: The dimension is not consistant with the initialization.\n")

        self.setBound(init_inputs, init_outputs, init_other, radius)
        self.setVarValue(inputs, outputs, other)
        self.TRF.RMConstraint.deactivate()
        self.objective.activate()
        self.setParam(inputs=init_inputs, RMParams=RMParams)
        self.TRF.RMConstraint.activate()

        return self.solveModel()

    def initialGeometry(self, numberOfInputs):
        """
        Generate geometry
        """
        conditionNumber, self.optimalPointSet, self.optimalMatrix = \
                generate_geometry(numberOfInputs)

    def regressRMParams(self, inputs, radius):
        """
        Calculate mathematical regression of RM near inputs with perturbation
        """
        val_cache = self.getEFInputValues()
        outputs = self.evaluateEF(inputs)
        RMParams = {}

        if (self.rmtype == RMType.linear):
            externalNodes = self.TRF.ef_variables.keys()
            # Set constant coefficient for each node
            for externalNode in externalNodes:
                index = self.TRF.ef_variables[externalNode]['index']
                RMParams[externalNode] = [outputs[index]]
                RMParams[externalNode]['linear'] = []
            # Set linear coefficients for each node
            for i in range(self.numberOfInputs):
                tempInputs = list(inputs)
                tempInputs[i] += radius
                tempOutputs = self.evaluateEF(tempInputs)
                for externalNode, value in zip(externalNodes, tempOutputs):
                    index = self.TRF.ef_variables[externalNode]['index']
                    scaled = (value - outputs[index]) / radius
                    RMParams[externalNode].append(scaled)
        elif (self.rmtype == RMType.quadratic):
            if self.optimalMatrix is None:
                self.initialGeometry(self.numberOfInputs)
            dimension = quadraticExpression(self.numberOfInputs)

            rhs = []
            for point in self.optimalPointSet[:-1]:
                y = self.evaluateEF(inputs + radius*point)
                rhs.append(y)
            rhs.append(outputs)

            coefficients = np.linalg.solve(self.optimalMatrix * radius,
                                           np.matrix(rhs))
            externalNodes = self.TRF.ef_variables.keys()
            for externalNode in externalNodes:
                RMParams[externalNode] = []
                # Set constant coefficient for each node
                RMParams[externalNode].append(coefficients[0, externalNode])
                # Set linear coefficients for each node
                RMParams[externalNode].append(coefficients[1:self.numberOfInputs+1, externalNode])
                # Set quadratic coefficients for each node
                RMParams.append(coefficients[self.numberOfInputs+1:, externalNode])

        self.setEFInputValues(val_cache)
        return RMParams, outputs
