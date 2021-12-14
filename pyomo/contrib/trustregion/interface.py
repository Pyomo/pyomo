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

from pyomo.common.dependencies import numpy as np
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Block, Param, VarList, Constraint,
    Objective, value, Set
    )
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import (identify_variables,
                                     ExpressionReplacementVisitor)
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


logger = logging.getLogger('pyomo.contrib.trustregion')


class EFReplacement(ExpressionReplacementVisitor):
    """
    NOTE: We use an empty substitution map. The EFs to be substituted are
          identified as part of exitNode.
    """
    def __init__(self, trfData, efSet):
        super().__init__(descend_into_named_expressions=True,
                         remove_named_expressions=False)
        self.trfData = trfData
        self.efSet = efSet

    def beforeChild(self, node, child, child_idx):
        # We want to capture all of the variables on the model
        descend, result = super().beforeChild(node, child, child_idx)
        # TODO: Make this match whatever identify_variables uses
        if not descend and result.__class__ not in native_types and result.is_variable_type():
            self.trfData.all_variables.add(result)
        return descend, result

    def exitNode(self, node, data):
        # This is where the replacement happens
        new_node = super().exitNode(node, data)
        if new_node.__class__ is not ExternalFunctionExpression:
            return new_node
        if self.efSet is not None and new_node._fcn not in self.efSet:
            # SKIP: efSet is provided and this node is not in it.
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

    def __init__(self, model, ext_fcn_surrogate_map_rule, config):
        self.original_model = model
        self.config = config
        self.model = self.original_model.clone()
        self.data = Block()
        self.model.add_component(unique_component_name(self.model, 'trf_data'),
                                 self.data)
        self.basis_expression_rule = ext_fcn_surrogate_map_rule
        self.efSet = None
        self.solver = SolverFactory(self.config.solver)
        # TODO: Provide an API for users to set this only to substitute
        # a subset of identified external functions.
        # Also rename to "efFilterSet" or something similar.

    def __exit__(self):
        pass

    def replaceEF(self, expr):
        """
        Replace an External Function.

        Arguments:
            expr  : a Pyomo expression. We will search this expression tree

        This function returns an expression after removing any
        ExternalFunction in the set efSet from the expression tree
        `expr` and replacing them with variables.
        New variables are declared on the `TRF` block.

        TODO: Future work - investigate direct substitution of basis or
        surrogate models using Expression objects instead of new variables.
        """
        return EFReplacement(self.data, self.efSet).walk_expression(expr)

    def _remove_ef_from_expr(self, component):
        expr = component.expr
        next_ef_id = len(self.data.ef_outputs)
        new_expr = self.replaceEF(expr)
        if new_expr is not expr:
            component.set_value(new_expr)
            new_output_vars = list(self.data.ef_outputs[i+1] for i in range(next_ef_id, len(self.data.ef_outputs)))
            for v in new_output_vars:
                self.data.basis_expressions[v] = \
                    self.basis_expression_rule(
                        component, self.data.truth_models[v])

    def replaceExternalFunctionsWithVariables(self):
        """
        Triggers the replacement of EFs with variables in expression trees
        """
        self.data.all_variables = ComponentSet()
        self.data.truth_models = ComponentMap()
        self.data.basis_expressions = ComponentMap()
        self.data.ef_inputs = {}
        self.data.ef_outputs = VarList()

        for con in self.model.component_data_objects(Constraint,
                                                     active=True):
            self._remove_ef_from_expr(con)

        objs = list(self.model.component_data_objects(Objective,
                                                      active=True))
        if len(objs) != 1:
            raise ValueError(
                "transformForTrustRegion: "
                "TrustRegion only supports models with a single active Objective.")
        self._remove_ef_from_expr(objs[0])

        for i in self.data.ef_outputs:
            self.data.ef_inputs[i] = \
                list(identify_variables(self.data.truth_models[self.data.ef_outputs[i]],
                                        include_fixed=False))

    def createConstraints(self):
        """
        Create constraints
        """
        b = self.data
        # This implements: y = b(w) from Yoshio/Biegler (2020)
        @b.Constraint(b.ef_outputs.index_set())
        def basis_constraints(b, i):
            ef_output_var = b.ef_outputs[i]
            return ef_output_var == b.basis_expressions[ef_output_var]
        b.basis_constraints.deactivate()

        b.INPUT_OUTPUT = Set(initialize=(
            (i, j) for i in b.ef_outputs.index_set()
            for j in range(len(b.ef_inputs[b.ef_outputs[i]]))
        ))
        b.basis_model_output = Param(b.ef_outputs.index_set(), mutable=True)
        b.grad_basis_model_output = Param(b.INPUT_OUTPUT, mutable=True)
        b.truth_model_output = Param(b.ef_outputs.index_set(), mutable=True)
        b.grad_truth_model_output = Param(b.INPUT_OUTPUT, mutable=True)
        # This implements: y = r_k(w)
        @b.Constraint(b.ef_outputs.index_set())
        def sm_constraint_basis(b, i):
            ef_output_var = b.ef_outputs[i]
            return ef_output_var == b.basis_expressions[ef_output_var] + \
                b.truth_model_output[i] - b.basis_model_output[i] + \
                sum((b.grad_truth_model_output[i, j]
                     - b.grad_basis_model_output[i, j])
                    * (w - b.ef_inputs[j])
                    for j, w in enumerate(b.ef_inputs[ef_output_var]))
        b.sm_constraint_basis.deactivate()

    def updateSurrogateModel(self):
        """
        Update relevant parameters
        """
        b = self.data
        for i, y in b.ef_outputs.items():
            b.basis_model_output[i] = value(b.basis_expressions[y])
            b.truth_model_output[i] = value(b.truth_models[y])
            # Basis functions are Pyomo expressions, in theory.
            # So this is possible to call
            gradBasis = differentiate(b.basis_expressions[y],
                                      wrt_list=b.ef_inputs[y])
            gradTruth = differentiate(b.truth_models[y],
                                      wrt_list=b.ef_inputs[y])
            for j, w in enumerate(b.ef_inputs[y]):
                b.grad_basis_model_output[i, j] = gradBasis[j]
                b.grad_truth_model_output[i, j] = gradTruth[j]

    def getCurrentEFValues(self):
        """
        Return current variable values
        """
        ans = ComponentMap()
        for output_var in self.data.ef_outputs:
            current_inputs = [value(input_var) for input_var in self.data.ef_inputs[output_var]]
            current_outputs = value(output_var)
            ans[output_var] = (current_inputs, current_outputs)
        return ans

    def calculateFeasibility(self):
        """
        Feasibility measure (theta(x)) is:
            || y - d(w) ||_1
        """
        b = self.data
        return sum(abs(value(y) - value(b.truth_model_output[i])) for i, y in b.ef_outputs.items())

    def initializeProblem(self):
        """
        Initializes appropriate constraints, values, etc. for TRF problem

        Returns
        -------
            objective_value : Initial objective
            feasibility : Initial feasibility measure

        STEPS:
            1. Create and solve PMP (eq. 3) and set equal to "x_0"
            2. Evaluate d(w_0)
            3. Evaluate initial feasibility measure (theta(x_0))
            4. Create initial SM (difference btw. low + high fidelity models)

        """
        self.replaceExternalFunctionsWithVariables()
        self.createConstraints()
        self.data.basis_constraints.activate()
        result, objective_value, _, _ = self.solveModel()
        self.data.basis_constraints.deactivate()
        self.updateSurrogateModel()
        feasibility = self.calculateFeasibility()
        self.data.sm_constraint_basis.activate()
        return objective_value, feasibility

    def solveModel(self):
        """
        Call the specified solver to solve the problem

        This also caches the previous values of the vars, just in case
        we need to access them later if a step is rejected
        """
        # TODO: Update this for new return type
        self.cached_inputs, self.cached_outputs = self.getCurrentValues()
        # TODO: need getModelVariables and setModelVariables
        results = self.solver.solve(self.model, keepfiles=self.config.keepfiles,
                                    tee=self.config.tee,
                                    load_solution=self.config.load_solution)

        if ((results.solver.status == SolverStatus.ok)
            and (results.solver.termination_condition ==
                 TerminationCondition.optimal)):
            self.model.solutions.load_from(results)
            # Find the step size norm between the values
            # TODO: Update this for new return type
            new_inputs, new_outputs = self.getCurrentValues()
            step_norm = np.linalg.norm(np.concatenate(self.cached_inputs - new_inputs,
                                                      self.cached_outputs - new_outputs),
                                       np.inf)
            feasibility = self.calculateFeasibility()
            for obj in self.model.component_data_objects(Objective, active=True):
                return True, obj(), step_norm, feasibility
        else:
            raise ArithmeticError('EXIT: Model solve failed with status {} and termination'
                                  ' condition(s) {}.'.format(str(results.solver.status),
                                                             str(results.solver.termination_condition)))

    def rejectStep(self):
        """
        If a step is rejected, we reset the model variables values back
        to their cached state - which we set in solveModel
        """
        # TODO: Change this to correctly reset the entire state space
        # TODO: need getModelVariables and setModelVariables
        # At the very beginning, we need to build a list of ALL vars in the model
        b = self.data
        for i, v in b.ef_outputs.items():
            b.ef_outputs[i].set_value(self.cached_outputs[i])
            for j, y in enumerate(b.ef_inputs[v]):
                y.set_value(self.cached_inputs[i][j])
