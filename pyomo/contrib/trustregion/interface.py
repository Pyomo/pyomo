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
    #    TRF.exfn_xvars : Dict (for which the keys are TRF.external_fcns)
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

        self.len_x = len(self.TRF.xvars)
        # Number of "glass box" variables
        self.len_z = len(self.TRF.zvars)
        # Number of external functions (externalNodes)
        self.len_y = len(self.TRF.y)

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

        allVariables = set(var for var in model.component_data_objects(Var))

        model.add_component(unique_component_name(model, 'tR'), TRF)
        TRF.y = VarList()
        TRF.x = VarList()
        TRF.constraints_ref_ef = ConstraintList()
        TRF.objective_ref_ef = ObjectiveList()
        TRF.exfn_xvars = {}
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

        # xT is an aggregated vector of variables [wT, yT, zT]
        TRF.xvars = []
        # Populate xvars
        seenVar = set()
        for externalNode in TRF.exfn_xvars.keys():
            for var in TRF.exfn_xvars[externalNode]['vars']:
                if id(var) not in seenVar:
                    seenVar.add(id(var))
                    TRF.xvars.append(var)

        # z is a vector of "glass box"/state/decision variables
        TRF.zvars = []
        # Remaining variables from allVariables are assigned to zvars
        for var in allVariables:
            if id(var) not in seenVar:
                seenVar.add(id(var))
                TRF.zvars.append(var)

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
        # We initialize y as 1s anyway right now
        y = np.ones(self.len_y, dtype=float)
        x = np.zeros(self.len_x, dtype=float)
        z = np.zeros(self.len_z, dtype=float)
        for i in range(0, self.len_x):
            x[i] = value(self.TRF.xvars[i])
        for i in range(0, self.len_z):
            z[i] = value(self.TRF.zvars[i])
        return x, y, z

    def createParam(self):
        """
        Create relevant parameters
        """
        ind_lx = RangeSet(0, self.len_x-1)
        self.TRF.ind_ly = RangeSet(0, self.len_y-1)
        ind_lz = RangeSet(0, self.len_z-1)
        self.TRF.x0 = Param(ind_lx,
                            mutable=True, default=0)
        self.TRF.y0 = Param(self.TRF.ind_ly,
                            mutable=True, default=0)
        self.TRF.z0 = Param(ind_lz,
                            mutable=True, default=0)
        if self.rmtype == RMType.linear:
            self.TRF.rmParam = Param(self.TRF.ind_ly,
                                     range(self.len_x + 1),
                                     mutable=True, default=0)
        elif self.rmtype == RMType.quadratic:
            self.TRF.rmParam = Param(self.TRF.ind_ly,
                                     range(quadraticExpression(self.len_x)),
                                     mutable=True, default=0)
        else:
            raise RuntimeError(
                "createParam: TrustRegion only supports linear and quadratic RM types.")

    def RMConstraint(self, model, externalNode):
        """
        Expression for use in the RM Constraint
        """
        variables = self.TRF.exfn_xvars[externalNode]['vars']
        idx = self.TRF.exfn_xvars[externalNode]['index']
        if self.rmtype == RMType.linear:
            # FIXME: This is wrong. variables[j] != index[j]
            # Original code:
            #    e1 = (model.plrom[i,0] + sum(model.plrom[i,j+1] * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(0, len(ind))))
            expr = (model.rmParam[idx, 0] +
                sum(model.rmParam[idx, j+1] * (model.xvars[variables[j]] -
                                                model.x0[variables[j]])
                    for j in range(0, len(variables))))
        elif self.rmtype == RMType.quadratic:
            expr = (model.rmParam[idx, 0] +
            sum(model.rmParam[idx, j+1] *
                (model.xvars[j] - model.x0[j])
                for j in range(0, self.len_x)))
            i = self.len_x + 1
            for j1 in range(self.len_x):
                for j2 in range(j1, self.len_x):
                    expr += ((model.xvars[j2] - model.x0[j2]) *
                             (model.xvars[j1] - model.x0[j1]) * model.rmParam[idx, i])
                    i += 1
        return expr

    def createRMConstraint(self):
        """
        Create appropriate constraints for RM
        """
        def RMConstraintRule(model, externalNode):
            idx = self.TRF.exfn_xvars[externalNode]['index']
            return model.y[idx] == self.RMConstraint(model, externalNode)
        self.TRF.RMConstraint = Constraint(self.TRF.ind_ly,
                                           rule=RMConstraintRule)

    def cacheBound(self):
        """
        Store the upper and lower bounds for each variable
        """
        self.TRF.xvar_lower = []
        self.TRF.xvar_upper = []
        self.TRF.zvar_lower = []
        self.TRF.zvar_upper = []
        for x in self.TRF.xvars:
            self.TRF.xvar_lower.append(x.lb)
            self.TRF.xvar_upper.append(x.up)
        for z in self.TRF.zvars:
            self.TRF.zvar_lower.append(z.lb)
            self.TRF.zvar_upper.append(z.up)

    def setParam(self, x0=None, RMParams=None):
        """
        Populate parameter values
        """
        if x0 is not None:
            self.TRF.x0.store_values(x0)
        if RMParams is not None:
            externalNodes = self.TRF.exfn_xvars.keys()
            for externalNode in externalNodes:
                idx = self.TRF.exfn_xvars[externalNode]['index']
                for j in range(len(RMParams[externalNode])):
                    self.TRF.rmParams[idx, j] = RMParams[externalNode][j]

    def setVarValue(self, x=None, y=None, z=None):
        """
        Populate variable values
        """
        if x is not None:
            if(len(x) != self.len_x):
                raise Exception(
                    "setVarValue: The dimension of x is not consistent.\n")
            self.TRF.xvars.set_values(x)
        if y is not None:
            if(len(y) != self.len_y):
                raise Exception(
                    "setVarValue: The dimension of y is not consistent.\n")
            self.TRF.y.set_values(y)
        if z is not None:
            if(len(z) != self.len_z):
                raise Exception(
                    "setVarValue: The dimension of z is not consistent.\n")
            self.TRF.zvars.set_values(z)

    def setBound(self, x0, y0, z0, radius):
        """
        Set bounds for variables
        """
        for i in range(0, self.len_x):
            self.TRF.xvars[i].setlb(maxIgnoreNone(x0[i] -
                                                  radius, self.TRF.xvar_lower[i]))
            self.TRF.xvars[i].setub(minIgnoreNone(x0[i] +
                                                  radius, self.TRF.xvar_upper[i]))
        for i in range(0, self.len_y):
            self.TRF.y[i].setlb(y0[i] - radius)
            self.TRF.y[i].setub(y0[i] + radius)
        for i in range(0, self.len_z):
            self.TRF.zvars[i].setlb(maxIgnoreNone(z0[i] -
                                                  radius, self.TRF.zvar_lower[i]))
            self.TRF.zvars[i].setub(minIgnoreNone(z0[i] +
                                                  radius, self.TRF.zvar_upper[i]))

    def externalNodeValues(self, externalNode, x):
        """
        Return values for x based on supplied externalNode
        """
        values = []
        variables = self.TRF.exfn_xvars[externalNode]['vars']
        for var in variables:
            values.append(x[variables.index(var)])
        return values
        

    def evaluateDx(self, x):
        """
        Evaluate values at external functions
        """
        ans = []
        externalNodes = self.TRF.exfn_xvars.keys()
        for externalNode in externalNodes:
            values = self.externalNodeValues(externalNode, x)
            ans.append(externalNode._fcn(*values))
        return np.array(ans)

    def evaluateObj(self, x, y, z):
        """
        Evaluate the objective
        """
        self.setVarValue(x=x, y=y, z=z)
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

    def TRSPk(self, x, y, z, x0, y0, z0, RMParams, radius):
        """
        Trust region subproblem solver

        Used for each iteration of the trust region filter method
        """
        if ((len(x) != self.len_x) or (len(y) != self.len_y)
            or (len(z) != self.len_z) or (len(x0) != self.len_x)
            or (len(y0) != self.len_y) or (len(z0) != self.len_z)):
            raise RuntimeError(
                "TRSP_k: The dimension is not consistant with the initialization.\n")

        self.setBound(x0, y0, z0, radius)
        self.setVarValue(x, y, z)
        self.TRF.RMConstraint.deactivate()
        self.objective.activate()
        self.setParam(x0=x0, RMParams=RMParams)
        self.TRF.RMConstraint.activate()

        return self.solveModel()

    def initialGeometry(self, len_x):
        """
        Generate geometry
        """
        conditionNumber, self.optimalPointSet, self.optimalMatrix = \
                generate_geometry(len_x)

    def buildRM(self, x, radius):
        """
        Builds reduced model (RM) near point x based on perturbation
        """
        y0 = self.evaluateDx(x)
        # RMParams has the format:
        #    {'externalNode' : [list of coefficient values]}
        RMParams = {}

        if (self.rmtype == RMType.linear):
            externalNodes = self.TRF.exfn_xvars.keys()
            for externalNode in externalNodes:
                index = self.TRF.exfn_xvars[externalNode]['index']
                values = self.externalNodeValues(externalNode, x)
                for i in range(0, len(values)):
                    tempValues = values
                    tempValues[i] = tempValues[i] + radius
                    y1 = externalNode._fcn(*tempValues)
                    RMParams[externalNode] = (y1 - y0[index]) / radius

        elif (self.rmtype == RMType.quadratic):
            if self.optimalMatrix is None:
                self.initialGeometry(self.len_x)
            dimension = quadraticExpression(self.len_x)
            rhs = []
            for point in self.optimalPointSet[:-1]:
                y = self.evaluateDx(x + radius*point)
                rhs.append(y)
            rhs.append(y0)

            coefficients = np.linalg.solve(self.optimalMatrix, np.matrix(rhs))
            externalNodes = self.TRF.exfn_xvars.keys()
            for externalNode in externalNodes:
                index = self.TRF.exfn_xvars[externalNode]['index']
                RMParams[externalNode] = []
                for d in range(0, dimension):
                    RMParams[externalNode].append(coefficients[d, index])
                for linearTerm in range(1, self.len_x + 1):
                    RMParams[externalNode][linearTerm] = \
                        RMParams[externalNode][linearTerm]/radius
                quadTermCounter= self.len_x + 1
                for lx1 in range(0, self.len_x):
                    for lx2 in range(lx1, self.len_x):
                        ###################################
                        # FIXME: It is unclear if this is
                        # the correct scaling.
                        # We need to revisit this.
                        # mmundt - Oct 4, 2021
                        ###################################
                        RMParams[externalNode][quadTermCounter] = \
                            RMParams[externalNode][quadTermCounter]/(radius)
                        quadTermCounter += 1
        return RMParams, y0
