#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import logging

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.trustregion.util import minIgnoreNone, maxIgnoreNone
from pyomo.core import (
    Block,
    Param,
    VarList,
    Constraint,
    Objective,
    value,
    Set,
    ExternalFunction,
    maximize,
    minimize,
)
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, check_optimal_termination


logger = logging.getLogger('pyomo.contrib.trustregion')


class EFReplacement(ExpressionReplacementVisitor):
    """
    This class is a subclass of ExpressionReplacementVisitor.
    It replaces an external function expression in an expression tree with a
    "holder" variable (recorded in a ComponentMap) and sets the initial value
    of the new node on the tree to that of the original node, if it can.

    NOTE: We use an empty substitution map. The EFs to be substituted are
          identified as part of exitNode.
    """

    def __init__(self, trfData, efSet):
        super().__init__(
            descend_into_named_expressions=True, remove_named_expressions=False
        )
        self.trfData = trfData
        self.efSet = efSet

    def beforeChild(self, node, child, child_idx):
        # We want to capture all of the variables on the model.
        # If we reject a step, we need to know all the vars to reset.
        descend, result = super().beforeChild(node, child, child_idx)
        if (
            not descend
            and result.__class__ not in native_types
            and result.is_variable_type()
        ):
            self.trfData.all_variables.add(result)
        return descend, result

    def exitNode(self, node, data):
        # This is where the replacement happens.
        new_node = super().exitNode(node, data)
        if new_node.__class__ is not ExternalFunctionExpression:
            return new_node
        if self.efSet is not None and new_node._fcn not in self.efSet:
            # SKIP: efSet is provided and this node is not in it.
            return new_node

        _output = self.trfData.ef_outputs.add()
        # Set the value for the new node to the evaluated value of the
        # original node, if possible.
        try:
            _output.set_value(value(node))
        except:
            _output.set_value(0)
        # Preserve the new node as a truth model.
        # self.TRF.truth_models is a ComponentMap.
        self.trfData.truth_models[_output] = new_node
        return _output


class TRFInterface(object):
    """
    Pyomo interface for Trust Region algorithm.
    """

    def __init__(self, model, decision_variables, ext_fcn_surrogate_map_rule, config):
        self.original_model = model
        tmp_name = unique_component_name(self.original_model, 'tmp')
        setattr(self.original_model, tmp_name, decision_variables)
        self.config = config
        self.model = self.original_model.clone()
        self.decision_variables = getattr(self.model, tmp_name)
        delattr(self.original_model, tmp_name)
        self.data = Block()
        self.model.add_component(
            unique_component_name(self.model, 'trf_data'), self.data
        )
        self.basis_expression_rule = ext_fcn_surrogate_map_rule
        self.efSet = None
        self.solver = SolverFactory(self.config.solver)
        # TODO: Provide an API for users to set this only to substitute
        # a subset of identified external functions.
        # Also rename to "efFilterSet" or something similar.

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
        """
        This method takes a component and looks at its expression.
        If the expression contains an external function (EF), a new expression
        with the EF replaced with a "holder" variable is added to the component
        and the basis expression for the new "holder" variable is updated.
        """
        expr = component.expr
        next_ef_id = len(self.data.ef_outputs)
        new_expr = self.replaceEF(expr)
        if new_expr is not expr:
            component.set_value(new_expr)
            new_output_vars = list(
                self.data.ef_outputs[i + 1]
                for i in range(next_ef_id, len(self.data.ef_outputs))
            )
            for v in new_output_vars:
                self.data.basis_expressions[v] = self.basis_expression_rule(
                    component, self.data.truth_models[v]
                )

    def replaceExternalFunctionsWithVariables(self):
        """
        This method sets up essential data objects on the new trf_data block
        on the model as well as triggers the replacement of external functions
        in expressions trees.

        Data objects created:
            self.data.all_variables : ComponentSet
                A set of all variables on the model, including "holder"
                variables from the EF replacement
            self.data.truth_models : ComponentMap
                A component map for replaced nodes that keeps track of
                the truth model for that replacement.
            self.data.basis_expressions : ComponentMap
                A component map for the Pyomo expressions for basis functions
                as they apply to each variable
            self.data.ef_inputs : Dict
                A dictionary that tracks the input variables for each EF
            self.data.ef_outputs : VarList
                A list of the "holder" variables which replaced the original
                External Function expressions
        """
        self.data.all_variables = ComponentSet()
        self.data.truth_models = ComponentMap()
        self.data.basis_expressions = ComponentMap()
        self.data.ef_inputs = {}
        self.data.ef_outputs = VarList()

        number_of_equality_constraints = 0
        for con in self.model.component_data_objects(Constraint, active=True):
            if con.lb == con.ub and con.lb is not None:
                number_of_equality_constraints += 1
            self._remove_ef_from_expr(con)

        self.degrees_of_freedom = (
            len(list(self.data.all_variables)) - number_of_equality_constraints
        )
        if self.degrees_of_freedom != len(self.decision_variables):
            raise ValueError(
                "replaceExternalFunctionsWithVariables: "
                "The degrees of freedom %d do not match the number of decision "
                "variables supplied %d."
                % (self.degrees_of_freedom, len(self.decision_variables))
            )

        for var in self.decision_variables:
            if var not in self.data.all_variables:
                raise ValueError(
                    "replaceExternalFunctionsWithVariables: "
                    f"The supplied decision variable {var.name} cannot "
                    "be found in the model variables."
                )

        self.data.objs = list(self.model.component_data_objects(Objective, active=True))
        # HACK: This is a hack that we will want to remove once the NL writer
        # has been corrected to not send unused EFs to the solver
        for ef in self.model.component_objects(ExternalFunction):
            ef.parent_block().del_component(ef)

        if len(self.data.objs) != 1:
            raise ValueError(
                "replaceExternalFunctionsWithVariables: "
                "TrustRegion only supports models with a single active Objective."
            )
        if self.data.objs[0].sense == maximize:
            self.data.objs[0].expr = -1 * self.data.objs[0].expr
            self.data.objs[0].sense = minimize
        self._remove_ef_from_expr(self.data.objs[0])

        for i in self.data.ef_outputs:
            self.data.ef_inputs[i] = list(
                identify_variables(
                    self.data.truth_models[self.data.ef_outputs[i]], include_fixed=False
                )
            )
        self.data.all_variables.update(self.data.ef_outputs.values())
        self.data.all_variables = list(self.data.all_variables)

    def createConstraints(self):
        """
        Create the basis constraint y = b(w) (equation 3) and the
        surrogate model constraint y = r_k(w) (equation 5)

        Both constraints are immediately deactivated after creation and
        are activated later as necessary.
        """
        b = self.data

        # This implements: y = b(w) from Yoshio/Biegler (2020)
        @b.Constraint(b.ef_outputs.index_set())
        def basis_constraint(b, i):
            ef_output_var = b.ef_outputs[i]
            return ef_output_var == b.basis_expressions[ef_output_var]

        b.basis_constraint.deactivate()

        b.INPUT_OUTPUT = Set(
            initialize=(
                (i, j)
                for i in b.ef_outputs.index_set()
                for j in range(len(b.ef_inputs[i]))
            )
        )
        b.basis_model_output = Param(b.ef_outputs.index_set(), mutable=True)
        b.grad_basis_model_output = Param(b.INPUT_OUTPUT, mutable=True)
        b.truth_model_output = Param(b.ef_outputs.index_set(), mutable=True)
        b.grad_truth_model_output = Param(b.INPUT_OUTPUT, mutable=True)
        b.value_of_ef_inputs = Param(b.INPUT_OUTPUT, mutable=True)

        # This implements: y = r_k(w)
        @b.Constraint(b.ef_outputs.index_set())
        def sm_constraint_basis(b, i):
            ef_output_var = b.ef_outputs[i]
            return ef_output_var == b.basis_expressions[
                ef_output_var
            ] + b.truth_model_output[i] - b.basis_model_output[i] + sum(
                (b.grad_truth_model_output[i, j] - b.grad_basis_model_output[i, j])
                * (w - b.value_of_ef_inputs[i, j])
                for j, w in enumerate(b.ef_inputs[i])
            )

        b.sm_constraint_basis.deactivate()

    def getCurrentDecisionVariableValues(self):
        """
        Return current decision variable values
        """
        decision_values = {}
        for var in self.decision_variables:
            decision_values[var.name] = value(var)
        return decision_values

    def updateDecisionVariableBounds(self, radius):
        """Update the TRSP_k decision variable bounds

        This corresponds to:

        .. math::
            || E^{-1} (u - u_k) || <= trust_radius

        We omit :math:`E^{-1}` because we assume that the users have
        correctly scaled their variables.

        """
        for var in self.decision_variables:
            var.setlb(
                maxIgnoreNone(
                    value(var) - radius, self.initial_decision_bounds[var.name][0]
                )
            )
            var.setub(
                minIgnoreNone(
                    value(var) + radius, self.initial_decision_bounds[var.name][1]
                )
            )

    def updateSurrogateModel(self):
        """
        The parameters needed for the surrogate model are the values of:
            b(w_k)      : basis_model_output
            d(w_k)      : truth_model_output
            grad b(w_k) : grad_basis_model_output
            grad d(w_k) : grad_truth_model_output
        """
        b = self.data
        for i, y in b.ef_outputs.items():
            b.basis_model_output[i] = value(b.basis_expressions[y])
            b.truth_model_output[i] = value(b.truth_models[y])
            # Basis functions are Pyomo expressions (in theory)
            gradBasis = differentiate(b.basis_expressions[y], wrt_list=b.ef_inputs[i])
            # These, however, are external functions
            gradTruth = differentiate(b.truth_models[y], wrt_list=b.ef_inputs[i])
            for j, w in enumerate(b.ef_inputs[i]):
                b.grad_basis_model_output[i, j] = gradBasis[j]
                b.grad_truth_model_output[i, j] = gradTruth[j]
                b.value_of_ef_inputs[i, j] = value(w)

    def getCurrentModelState(self):
        """
        Return current state of all model variables.
        This is necessary if we need to reject a step and move backwards.
        """
        return list(value(v, exception=False) for v in self.data.all_variables)

    def calculateFeasibility(self):
        """
        Feasibility measure (theta(x)) is:
            || y - d(w) ||_1
        """
        b = self.data
        return sum(
            abs(value(y) - value(b.truth_models[y])) for i, y in b.ef_outputs.items()
        )

    def calculateStepSizeInfNorm(self, original_values, new_values):
        """
        Taking original and new values, calculate the step-size norm ||s_k||:
            || u - u_k ||_inf

        We assume that the user has correctly scaled their variables.
        """
        original_vals = []
        new_vals = []
        for var, val in original_values.items():
            original_vals.append(val)
            new_vals.append(new_values[var])
        return max([abs(new - old) for new, old in zip(new_vals, original_vals)])

    def initializeProblem(self):
        """
        Initializes appropriate constraints, values, etc. for TRF problem

        Returns
        -------
            objective_value : Initial objective
            feasibility     : Initial feasibility measure

        STEPS:
            1. Create and solve PMP (eq. 3) and set equal to "x_0"
            2. Evaluate d(w_0)
            3. Evaluate initial feasibility measure (theta(x_0))
            4. Create initial SM (difference btw. low + high fidelity models)

        """
        self.replaceExternalFunctionsWithVariables()
        self.initial_decision_bounds = {}
        for var in self.decision_variables:
            self.initial_decision_bounds[var.name] = [var.lb, var.ub]
        self.createConstraints()
        self.data.basis_constraint.activate()
        objective_value, _, _ = self.solveModel()
        self.data.basis_constraint.deactivate()
        self.updateSurrogateModel()
        feasibility = self.calculateFeasibility()
        self.data.sm_constraint_basis.activate()
        return objective_value, feasibility

    def solveModel(self):
        """
        Call the specified solver to solve the problem.

        Returns
        -------
            self.data.objs[0] : Current objective value
            step_norm         : Current step size inf norm
            feasibility       : Current feasibility measure

        This also caches the previous values of the vars, just in case
        we need to access them later if a step is rejected
        """
        current_decision_values = self.getCurrentDecisionVariableValues()
        self.data.previous_model_state = self.getCurrentModelState()
        results = self.solver.solve(
            self.model, keepfiles=self.config.keepfiles, tee=self.config.tee
        )

        if not check_optimal_termination(results):
            raise ArithmeticError(
                'EXIT: Model solve failed with status {} and termination'
                ' condition(s) {}.'.format(
                    str(results.solver.status),
                    str(results.solver.termination_condition),
                )
            )

        self.model.solutions.load_from(results)
        new_decision_values = self.getCurrentDecisionVariableValues()
        step_norm = self.calculateStepSizeInfNorm(
            current_decision_values, new_decision_values
        )
        feasibility = self.calculateFeasibility()
        return self.data.objs[0](), step_norm, feasibility

    def rejectStep(self):
        """
        If a step is rejected, we reset the model variables values back
        to their cached state - which we set in solveModel
        """
        for var, val in zip(self.data.all_variables, self.data.previous_model_state):
            var.set_value(val, skip_validation=True)
