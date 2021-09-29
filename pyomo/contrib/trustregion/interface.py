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
from math import inf

from pyomo.common.collections import ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Block, Var, Param, VarList, ConstraintList, Constraint, Objective,
    RangeSet, value, ConcreteModel, Reals, sqrt, minimize, maximize,
    ObjectiveList
)
from pyomo.core.expr import current as EXPR
from pyomo.core.base.external import PythonCallbackFunction
from pyomo.core.base.numvalue import nonpyomo_leaf_types
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.contrib.trustregion.geometry import generate_geometry
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
    #    { 'node': {'index': #, 'vars': [vars]}}
    pass


class PyomoInterface(object):
    """
    Initialize with a pyomo model m.
    This is used in TrustRegionMethod.py, same requirements for m apply

    m is reformulated into form for use in TRF algorithm

    Specified ExternalFunction() objects are replaced with new variables
    All new attributes (including these variables) are stored on block
    "tR"


    Note: quadratic RM is messy, uses full dimension of x variables.
          clean up later.

    """
    countDx = -1
    rmtype = RMType.linear

    def __init__(self, m, efList, config):

        self.config = config
        self.model = m;
        self.TRF = self.transformForTrustRegion(self.model, efList)

        self.lengthx = len(self.TRF.xvars)
        self.lengthz = len(self.TRF.zvars)
        self.lengthy = len(self.TRF.y)

        self.createParam()
        self.createRMConstraint()
        self.createCompCheckObjective()
        self.cacheBound()

        self.geometry = None
        self.pset = None

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

        exfn_vars = [v for sublist in TRF.exfn_vars for v in sublist]
        # xT is an aggregated vector of variables [wT, yT, zT]
        TRF.xvars = []
        # Populate xvars
        seenVar = set()
        for var in exfn_vars:
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

        # Dictionary of vars and their indices
        self.exfn_xvars_index = {}
        for var in exfn_vars:
            for x in TRF.xvars:
                if (id(var) == id(x)):
                    self.exfn_xvars_index[var] = TRF.xvars.index(x)

        TRF.external_fcns = list(TRF.exfn_vars.keys())
        return TRF
            
    def reactivateObjects(self, model):
        """
        Reactivate objects that were deactivated as a result
        of the transformForTrustRegion method
        """
        for obj in model.TRF.deactivated_objects:
            obj.reactivate()

    def getInitialValue(self):
        """
        Initialize values into numpy array
        """
        # We initialize y as 1s anyway right now
        y = np.ones(self.lengthy, dtype=float)
        x = np.zeros(self.lengthx, dtype=float)
        z = np.zeros(self.lengthz, dtype=float)
        for i in range(0, self.lengthx):
            x[i] = value(self.TRF.xvars[i])
        for i in range(0, self.lengthz):
            z[i] = value(self.TRF.zvars[i])
        return x, y, z

    def createParam(self):
        """
        Create relevant parameters
        """
        self.TRF.ind_lx = RangeSet(0, self.lengthx-1)
        self.TRF.ind_ly = RangeSet(0, self.lengthy-1)
        self.TRF.ind_lz = RangeSet(0, self.lengthz-1)
        self.TRF.x0 = Param(self.TRF.ind_lx,
                            mutable=True, default=0)
        self.TRF.y0 = Param(self.TRF.ind_ly,
                            mutable=True, default=0)
        self.TRF.z0 = Param(self.TRF.ind_lz,
                            mutable=True, default=0)
        if self.rmtype == RMType.linear:
            self.TRF.rmParam = Param(self.TRF.ind_ly,
                                  range(self.lengthx + 1),
                                  mutable=True, default=0)
        elif self.rmtype == RMType.quadratic:
            self.TRF.rmParam = Param(self.TRF.ind_ly,
                                     range(int((self.lengthx*self.lengthx
                                                + self.lengthx*3)/2. + 1)),
                                     mutable=True, default=0)
        else:
            raise RuntimeError(
                "createParam: TrustRegion only support linear and quadratic RM types.")

    def RMConstraint(self, model, node):
        """
        Expression for use in the RM Constraint
        """
        variables = self.exfn_xvars_index[node]['vars']
        idx = self.exfn_xvars_index[node]['index']
        if self.rmtype == RMType.linear:
            expr = (model.rmParam[idx, 0] +
                sum(model.rmParam[idx, j+1] * (model.xvars[variables[j]] -
                                                model.x0[variables[j]])
                    for j in range(0, len(variables))))
        elif self.rmtype == RMType.quadratic:
            expr = (model.rmParam[idx, 0] +
            sum(model.rmParam[idx, j+1] *
                (model.xvars[j] - model.x0[j])
                for j in range(0, self.lengthx)))
            i = self.lengthx + 1
            for j1 in range(self.lengthx):
                for j2 in range(j1, self.lengthx):
                    expr += ((model.xvars[j2] - model.x0[j2]) *
                             (model.xvars[j1] - model.x0[j1]) * model.rmParam[idx, i])
                    i += 1
        return expr

    def createRMConstraint(self):
        """
        Create appropriate constraints for RM
        """
        def RMConstraintRule(model, node):
            idx = self.exfn_xvars_index[node]['index']
            return model.y[idx+1] == self.RMConstraint(model, node)
        self.TRF.RMConstraint = Constraint(self.TRF.ind_ly,
                                                 rule=RMConstraintRule)

    def cacheBounds(self):
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

    def setParams(self, x0=None, RMParams=None):
        """
        Populate parameter values
        """
        if x0 is not None:
            self.TRF.x0.store_values(x0)
        if RMParams is not None:
            for i in range(self.lengthy):
                for j in range(len(RMParams[i])):
                    self.TRF.rmParams[i, j] = RMParams[i][j]





