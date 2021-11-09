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

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Block, Var, Param, ExternalFunction,
    VarList, ConstraintList, Constraint,
    Objective, ObjectiveList, value
    )
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import (identify_variables,
                                     ExpressionReplacementVisitor)
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


logger = logging.getLogger('pyomo.contrib.trustregion')


class EFReplacement(ExpressionReplacementVisitor):
    def __init__(self, trfData, efSet):
        super().__init__(descend_into_named_expressions=True,
                         remove_named_expressions=False)
        self.trfData = trfData
        self.efSet = efSet

    def exitNode(self, node, data):
        # This is where the replacement happens
        new_node = super().exitNode(node, data)
        if new_node.__class__ is not ExternalFunctionExpression:
            return new_node
        if id(new_node._fcn) not in self.efSet:
            return new_node
        
        _output = self.trfData.ef_outputs.add()
        # Preserve the new node as a truth model
        # self.TRF.truth_models is a ComponentMap
        self.trfData.truth_models[_output] = new_node
        return _output


class TRFInterface(object):
    """
    Pyomo interface for Trust Region algorithm.
    """

    def __init__(self, model, config, ext_fcn_surrogate_map_rule):
        self.original_model = model
        self.config = config
        self.model = self.original_model.clone()
        self.data = Block()
        self.model.add_component(unique_component_name(model, 'trf_data'),
                                 self.data)
        self.basis_expression_rule = ext_fcn_surrogate_map_rule

    def __exit__(self):
        pass

    def replaceEF(self, expr, TRF, efSet):
        """
        Replace an External Function.

        Arguments:
            expr  : a Pyomo expression. We will search this expression tree
            TRF   : a Pyomo Block. We will add replacement vars on this Block
            efSet : the (Pyomo) set of external functions for which we
                    will use TRF method

        This function returns an expression after removing any
        ExternalFunction in the set efSet from the expression tree
        `expr`. New variables are declared on the `TRF` block and replace
        the external function.
        """
        return EFReplacement(TRF, efSet).walk_expression(expr)

    def _remove_ef_from_expr(self, component, efSet):
        expr = component.expr
        new_expr = self.replaceEF(expr, self.data, efSet)
        if new_expr is not expr:
            component.set_value(new_expr)
            new_output_vars = [self.data.ef_outputs[i] for i in
                               range(len(self.data.basis_expressions)+1,
                                     len(self.data.ef_outputs)+1)]
            for v in new_output_vars:
                self.data.basis_expressions[v] = \
                    self.basis_expression_rule(
                        component, self.data.truth_models[v])

    def transformForTrustRegion(self):
        efSet = set(self.model.component_data_objects(ExternalFunction))
        self.data.truth_models = ComponentMap()
        self.data.basis_expressions = ComponentMap() # Maybe
        self.data.ef_inputs = ComponentMap()
        self.data.ef_outputs = VarList()

        for con in self.model.component_data_objects(Constraint,
                                                     active=True):
            self._remove_ef_from_expr(con, efSet)

        objs = list(self.model.component_data_objects(Objective,
                                                      active=True))
        if len(objs) != 1:
            raise RuntimeError(
                "transformForTrustRegion: "
                "TrustRegion only supports models with a single active Objective.")
        self._remove_ef_from_expr(objs[0], efSet)

        for v in self.data.ef_outputs:
            self.data.ef_inputs[v] = \
                list(identify_variables(self.data.truth_models[v],
                                        include_fixed=False))

        # Process Model Problem (3) from Yoshio/Biegler (2020)
        self.pmp = self.model.clone()
        # Trust Region Subproblem (TRSP) (4) from Yoshio/Biegler (2020)
        self.trsp = self.model.clone()

    def solveModel(self, m, input_vars, output_vars):
        """
        Call the specified solver to solve the problem
        """
        solver = SolverFactory(self.config.solver)
        results = solver.solve(m, keepfiles=self.config.keepfiles,
                               tee=self.config.tee,
                               load_solution=self.config.load_solution)

        if ((results.solver.status == SolverStatus.ok)
            and (results.solver.termination_condition ==
                 TerminationCondition.optimal)):
            m.solutions.load_from(results)
            # TODO: Add results back to model here
            for obj in m.component_data_objects(Objective, active=True):
                return True, obj()
        else:
            print("Warning: Solver Status: %s" 
                  % str(results.solver.status))
            print("Termination Conditions: %s" 
                  % str(results.solver.termination_condition))
            return False, 0

    def createComponents(self):
        """
        Create required components for the model(s)
        """
        pass

    def setInitialValues(self):
        """
        Set initial values for first iteration
        """
        # We want to set x0 and d(w0) here
        # x0 : solution to the process model problem (3)
        # d(w0) : solution of high-fidelity model at initial inputs
        pass

    def createConstraints(self):
        """
        Create constraints
        """
        data_name = self.data.name

        pmp_data = self.pmp.find_component(data_name)
        # This implements: y = b(w) from Yoshio/Biegler (2020)
        @pmp_data.Constraint(pmp_data.ef_outputs.index_set())
        def basis_constraints(b, i):
            ef_output_var = b.ef_outputs[i]
            return ef_output_var == b.basis_expressions[ef_output_var]

        trsp_data = self.trsp.find_component(data_name)
        # This implements: y = r_k(w)
        @trsp_data.Constraint(trsp_data.ef_outputs.index_set())
        def sm_constraint_basis(b, i):
            ef_output_var = b.ef_outputs[i]
            return ef_output_var == b.basis_expressions[ef_output_var] # + \
                # Other junk from Eq 5

    def initializeProblem(self):
        """
        Initializes appropriate constraints, values, etc. for TRF problem

        Returns
        -------
            obj : Initial objective
            feasibility : Initial feasibility measure
            SM : Initial surrogate model
        """
        pass

    def buildSM(self):
        """
        Build a surrogate model
        """
        pass

    def TRSP(self):
        """
        Solve Trust Region Subproblem
        """
        pass

    def reset(self):
        """
        Reset problem for next iteration
        """
        pass
