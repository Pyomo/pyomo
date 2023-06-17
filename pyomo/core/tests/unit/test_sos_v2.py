# ******************************************************************************
# ******************************************************************************

import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers

solver_name = "scip"

solver_available = bool(check_available_solvers(solver_name))

# *****************************************************************************
# *****************************************************************************


class TestSOS(unittest.TestCase):
    @unittest.skipIf(not solver_available, "The solver is not available.")
    def test_nonindexedsos(self, show_output: bool = False):
        "Test non-indexed SOS using a single pyomo Var component."

        opt = pyo.SolverFactory(solver_name)

        # *********************************************************************

        # non-indexed SOS constraints.....

        test_vectors = [
            # sos, expect. result, absolute tolerance, use rule parameter, case
            (1, 0.04999999999999999, 1e-3, True, 0),
            (1, 0.04999999999999999, 1e-3, False, 0),
            (2, -0.07500000000000001, 1e-3, True, 0),
            (2, -0.07500000000000001, 1e-3, False, 0),
            (1, 0.04999999999999999, 1e-3, True, 1),
            (1, 0.04999999999999999, 1e-3, False, 1),
            (2, -0.07500000000000001, 1e-3, True, 1),
            (2, -0.07500000000000001, 1e-3, False, 1),
            (1, 0.04999999999999999, 1e-3, True, 2),
            (1, 0.04999999999999999, 1e-3, False, 2),
            (2, -0.07500000000000001, 1e-3, True, 2),
            (2, -0.07500000000000001, 1e-3, False, 2),
            (1, 0.04999999999999999, 1e-3, True, 3),
            (1, 0.04999999999999999, 1e-3, False, 3),
            (2, -0.07500000000000001, 1e-3, True, 3),
            (2, -0.07500000000000001, 1e-3, False, 3),
            (1, 0.04999999999999999, 1e-3, True, 4),
            (1, 0.04999999999999999, 1e-3, False, 4),
            (2, -0.07500000000000001, 1e-3, True, 4),
            (2, -0.07500000000000001, 1e-3, False, 4),
            (1, 0.04999999999999999, 1e-3, True, 5),
            (1, 0.04999999999999999, 1e-3, False, 5),
            (2, -0.07500000000000001, 1e-3, True, 5),
            (2, -0.07500000000000001, 1e-3, False, 5),
            (1, 0.04999999999999999, 1e-3, True, 6),
            (1, 0.04999999999999999, 1e-3, False, 6),
            (2, -0.07500000000000001, 1e-3, True, 6),
            (2, -0.07500000000000001, 1e-3, False, 6),
            # trigger the error
            (1, 0.04999999999999999, 1e-3, True, 7),
            (1, 0.04999999999999999, 1e-3, False, 7),
            (2, -0.07500000000000001, 1e-3, True, 7),
            (2, -0.07500000000000001, 1e-3, False, 7),
        ]

        for sos, exp_res, abs_tol, use_rule, case in test_vectors:
            try:
                model = problem_nonindexed_sosn(case=case, n=sos, use_rule=use_rule)
            except NotImplementedError:
                continue

            opt.solve(model, tee=show_output)

            assert len(model.mysos) != 0

            assert math.isclose(pyo.value(model.OBJ), exp_res, abs_tol=abs_tol)

    @unittest.skipIf(not solver_available, "The solver is not available.")
    def test_indexedsos(self, show_output: bool = False):
        "Test indexed SOS using a single pyomo Var component."

        opt = pyo.SolverFactory(solver_name)

        # *********************************************************************

        # indexed SOS constraints.....

        test_vectors = [
            # sos, expect. result, absolute tolerance, use rule parameter, case
            (1, -7.5000000000e-02, 1e-3, True, 0),
            (1, -7.5000000000e-02, 1e-3, False, 0),
            (2, 1.1, 1e-3, True, 0),
            (2, 1.1, 1e-3, False, 0),
            (1, -7.5000000000e-02, 1e-3, True, 1),
            (1, -7.5000000000e-02, 1e-3, False, 1),
            (2, 1.1, 1e-3, True, 1),
            (2, 1.1, 1e-3, False, 1),
            (1, -7.5000000000e-02, 1e-3, True, 2),
            (1, -7.5000000000e-02, 1e-3, False, 2),
            (2, 1.1, 1e-3, True, 2),
            (2, 1.1, 1e-3, False, 2),
            (1, -7.5000000000e-02, 1e-3, True, 3),
            (1, -7.5000000000e-02, 1e-3, False, 3),
            (2, 1.1, 1e-3, True, 3),
            (2, 1.1, 1e-3, False, 3),
            # trigger the error
            (1, -7.5000000000e-02, 1e-3, True, 4),
            (1, -7.5000000000e-02, 1e-3, False, 4),
            (2, 1.1, 1e-3, True, 4),
            (2, 1.1, 1e-3, False, 4),
        ]

        for sos, exp_res, abs_tol, use_rule, case in test_vectors:
            try:
                model = problem_indexed_sosn(case=case, n=sos, use_rule=use_rule)
            except NotImplementedError:
                continue

            problem = model.create_instance()

            opt.solve(problem, tee=show_output)

            assert len(problem.mysos) != 0

            assert math.isclose(pyo.value(problem.OBJ), exp_res, abs_tol=abs_tol)

        # *********************************************************************

    @unittest.skipIf(not solver_available, "The solver is not available.")
    def test_nonindexedsos_multivar(self, show_output: bool = False):
        "Test non-indexed SOS made up of different Var components."

        opt = pyo.SolverFactory(solver_name)

        # *********************************************************************

        test_vectors = [
            # sos, expected result, absolute tolerance
            (1, 0.125, 1e-3),
            (2, -0.07500000000000001, 1e-3),
        ]

        for sos, exp_res, abs_tol in test_vectors:
            try:
                model = problem_nonindexed_sosn_different_vars(n=sos)
            except NotImplementedError:
                continue

            problem = model.create_instance()

            opt.solve(problem, tee=show_output)

            assert len(problem.mysos) != 0

            assert math.isclose(pyo.value(problem.OBJ), exp_res, abs_tol=abs_tol)

        # *********************************************************************

    @unittest.skipIf(not solver_available, "The solver is not available.")
    def test_indexedsos_multivar(self, show_output: bool = False):
        "Test indexed SOS made up of different Var components."

        opt = pyo.SolverFactory(solver_name)

        # *********************************************************************

        test_vectors = [
            # sos, expected result, absolute tolerance
            (1, -7.5000000000e-02, 1e-3),
            (2, 1.1, 1e-3),
        ]

        for sos, exp_res, abs_tol in test_vectors:
            try:
                model = problem_indexed_sosn_different_vars(n=sos)
            except NotImplementedError:
                continue

            problem = model.create_instance()

            opt.solve(problem, tee=show_output)

            assert len(problem.mysos) != 0

            assert math.isclose(pyo.value(problem.OBJ), exp_res, abs_tol=abs_tol)

        # *********************************************************************


# *****************************************************************************
# *****************************************************************************


def problem_nonindexed_sosn_different_vars(n: int = 1):
    # concrete model

    model = pyo.ConcreteModel()
    model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0, 40))

    model.A = pyo.Set(initialize=[1, 2, 4, 6])
    model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(
        expr=(
            1 * model.x[1]
            + 2 * model.y[1]
            + 3 * model.y[2]
            + -0.1 * model.y[4]
            + 0.5 * model.y[6]
        )
    )

    model.ConstraintY1_ub = pyo.Constraint(expr=model.y[1] <= 2)
    model.ConstraintY2_ub = pyo.Constraint(expr=model.y[2] <= 2)
    model.ConstraintY4_ub = pyo.Constraint(expr=model.y[4] <= 2)
    model.ConstraintY6_ub = pyo.Constraint(expr=model.y[6] <= 2)
    model.ConstraintYmin = pyo.Constraint(
        expr=(model.x[1] + model.y[1] + model.y[2] + model.y[6] >= 0.25)
    )

    def rule_mysos(m):
        var_list = [m.x[a] for a in m.x]
        var_list.extend([m.y[a] for a in m.A])
        weight_list = [i + 1 for i in range(len(var_list))]
        return (var_list, weight_list)

    model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

    return model


# *****************************************************************************
# *****************************************************************************


def problem_indexed_sosn_different_vars(n: int = 1):
    # abstract model

    model = pyo.AbstractModel()
    model.E = pyo.Set(initialize=[1, 2])

    model.A = pyo.Set(initialize=[1, 2, 3, 5, 6])
    model.B = pyo.Set(initialize=[2, 4])
    model.x = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, 40))
    model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    model.param_cx = pyo.Param(model.E, initialize={1: 1, 2: 1.5})
    model.param_cy = pyo.Param(model.A, initialize={1: 2, 2: 3, 3: -0.1, 5: 0.5, 6: 4})

    def obj_f(m):
        return sum(m.param_cx[e] * m.x[e] for e in m.E) + sum(
            m.param_cy[a] * m.y[a] for a in m.A
        )

    model.OBJ = pyo.Objective(rule=obj_f)

    def constr_ya_lb(m, a):
        return m.y[a] <= 2

    model.ConstraintYa_lb = pyo.Constraint(model.A, rule=constr_ya_lb)

    def constr_y_lb(m):
        return m.x[1] + m.x[2] + m.y[1] + m.y[2] + m.y[5] + m.y[6] >= 0.25

    model.ConstraintY_lb = pyo.Constraint(rule=constr_y_lb)

    if n == 2:
        # force the second SOS2 to have two non-zero variables
        def constr_y2_lb(m):
            return (
                # m.x[1]+
                # m.y[1]+
                m.y[2] + m.y[5] + m.y[6]
                >= 2.1
            )

        model.ConstraintY2_lb = pyo.Constraint(rule=constr_y2_lb)

    # with weights, using a Set and a Param

    model.mysosindex_x = pyo.Set(model.B, initialize={2: [1], 4: [2]})
    model.mysosindex_y = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})
    model.mysosweights_x = pyo.Param(
        model.E,  # model.A or a subset that covers all relevant members
        initialize={1: 4, 2: 8},  # weights define adjacency
    )
    model.mysosweights_y = pyo.Param(
        model.A,  # model.A or a subset that covers all relevant members
        initialize={1: 25.0, 3: 18.0, 2: 3, 5: 7, 6: 10},  # weights define adjacency
    )

    def rule_mysos(m, b):
        var_list = [m.x[e] for e in m.mysosindex_x[b]]
        var_list.extend([m.y[a] for a in m.mysosindex_y[b]])

        weight_list = [m.mysosweights_x[e] for e in m.mysosindex_x[b]]
        weight_list.extend([m.mysosweights_y[a] for a in m.mysosindex_y[b]])

        return (var_list, weight_list)

    model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)

    return model


# *****************************************************************************
# *****************************************************************************


def problem_nonindexed_sosn(case: int == 0, n: int = 1, use_rule: bool = False):
    # concrete model

    model = pyo.ConcreteModel()
    model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0, 40))

    model.A = pyo.Set(initialize=[1, 2, 4, 6])
    model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(
        expr=(
            1 * model.x[1]
            + 2 * model.y[1]
            + 3 * model.y[2]
            + -0.1 * model.y[4]
            + 0.5 * model.y[6]
        )
    )

    model.ConstraintY1_ub = pyo.Constraint(expr=model.y[1] <= 2)
    model.ConstraintY2_ub = pyo.Constraint(expr=model.y[2] <= 2)
    model.ConstraintY4_ub = pyo.Constraint(expr=model.y[4] <= 2)
    model.ConstraintY6_ub = pyo.Constraint(expr=model.y[6] <= 2)
    model.ConstraintYmin = pyo.Constraint(
        expr=(model.x[1] + model.y[1] + model.y[2] + model.y[6] >= 0.25)
    )

    if case == 0:
        if use_rule:

            def rule_mysos(m):
                return ([m.y[a] for a in m.A], [i + 1 for i, _ in enumerate(m.A)])

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            # index is not provided, nor are the weights
            model.mysos = pyo.SOSConstraint(var=model.y, sos=n)

    elif case == 1:
        # no weights, but use a list

        index = [2, 4, 6]

        if use_rule:

            def rule_mysos(m):
                return ([m.y[a] for a in index], [i + 1 for i, _ in enumerate(index)])

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(var=model.y, index=index, sos=n)

    elif case == 2:
        # no weights, but use pyo.Set component (has to be part of the model)

        model.mysosindex = pyo.Set(initialize=[2, 4, 6], within=model.A)

        if use_rule:

            def rule_mysos(m):
                return (
                    [m.y[a] for a in m.mysosindex],
                    [i + 1 for i, _ in enumerate(m.mysosindex)],
                )

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(var=model.y, index=model.mysosindex, sos=n)

    elif case == 3:
        # with weights, using a list and a dict

        index = [2, 4, 6]
        weights = {2: 25.0, 4: 18.0, 6: 22}

        if use_rule:

            def rule_mysos(m):
                return [m.y[a] for a in index], [weights[a] for a in index]

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                var=model.y, index=index, weights=weights, sos=n
            )

    elif case == 4:
        # with weights, using a set and a param

        model.mysosindex = pyo.Set(initialize=[2, 4, 6], within=model.A)
        model.mysosweights = pyo.Param(
            model.mysosindex, initialize={2: 25.0, 4: 18.0, 6: 22}
        )

        if use_rule:

            def rule_mysos(m):
                return (
                    [m.y[a] for a in m.mysosindex],
                    [m.mysosweights[a] for a in m.mysosindex],
                )

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                var=model.y, index=model.mysosindex, weights=model.mysosweights, sos=n
            )

    elif case == 5:
        # index is not provided, but the weights are provided as a dict

        weights = {1: 3, 2: 25.0, 4: 18.0, 6: 22}

        if use_rule:

            def rule_mysos(m):
                return ([m.y[a] for a in m.y], [weights[a] for a in m.y])

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                var=model.y,
                sos=n,
                weights=weights,
            )

    elif case == 6:
        # index is not provided, but the weights are provided as a dict

        model.mysosweights = pyo.Param(
            [1, 2, 4, 6], initialize={1: 3, 2: 25.0, 4: 18.0, 6: 22}
        )

        if use_rule:

            def rule_mysos(m):
                return ([m.y[a] for a in m.y], [m.mysosweights[a] for a in m.y])

            model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                var=model.y,
                sos=n,
                weights=model.mysosweights,
            )

    else:
        raise NotImplementedError

    return model


# *****************************************************************************
# *****************************************************************************


def problem_indexed_sosn(case: int == 0, n: int = 1, use_rule: bool = False):
    # abstract model

    model = pyo.AbstractModel()
    model.E = pyo.Set(initialize=[1])

    model.A = pyo.Set(initialize=[1, 2, 3, 5, 6])
    model.B = pyo.Set(initialize=[2, 4])
    model.x = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, 40))
    model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)

    model.param_cx = pyo.Param(model.E, initialize={1: 1})
    model.param_cy = pyo.Param(model.A, initialize={1: 2, 2: 3, 3: -0.1, 5: 0.5, 6: 4})

    def obj_f(m):
        return sum(m.param_cx[e] * m.x[e] for e in m.E) + sum(
            m.param_cy[a] * m.y[a] for a in m.A
        )

    model.OBJ = pyo.Objective(rule=obj_f)

    def constr_ya_lb(m, a):
        return m.y[a] <= 2

    model.ConstraintYa_lb = pyo.Constraint(model.A, rule=constr_ya_lb)

    def constr_y_lb(m):
        return m.x[1] + m.y[1] + m.y[2] + m.y[5] + m.y[6] >= 0.25

    model.ConstraintY_lb = pyo.Constraint(rule=constr_y_lb)

    if n == 2:
        # force the second SOS2 to have two non-zero variables
        def constr_y2_lb(m):
            return (
                # m.x[1]+
                # m.y[1]+
                m.y[2] + m.y[5] + m.y[6]
                >= 2.1
            )

        model.ConstraintY2_lb = pyo.Constraint(rule=constr_y2_lb)

    if case == 0:
        # with index, no weights, using a dict

        index = {2: [1, 3], 4: [2, 5, 6]}

        if use_rule:

            def rule_mysos(m, b):
                return (
                    [m.y[a] for a in index[b]],
                    [i + 1 for i, _ in enumerate(index[b])],
                )

            model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(model.B, var=model.y, sos=n, index=index)

    elif case == 1:
        # with index, no weights, using a Set object

        model.mysosindex = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})

        if use_rule:

            def rule_mysos(m, b):
                return (
                    [m.y[a] for a in m.mysosindex[b]],
                    [i + 1 for i, _ in enumerate(m.mysosindex[b])],
                )

            model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                model.B, var=model.y, sos=n, index=model.mysosindex
            )

    elif case == 2:
        # with weights, provided using a set and a dict

        index = {2: [1, 3], 4: [2, 5, 6]}
        # the weights define adjacency
        weights = {1: 25.0, 3: 18.0, 2: 3, 5: 7, 6: 10}

        if use_rule:

            def rule_mysos(m, b):
                return ([m.y[a] for a in index[b]], [weights[a] for a in index[b]])

            model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                model.B, var=model.y, sos=n, index=index, weights=weights
            )

    elif case == 3:
        # with weights, using a Set and a Param

        model.mysosindex = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})
        model.mysosweights = pyo.Param(
            model.A,  # model.A or a subset that covers all relevant members
            initialize={
                1: 25.0,
                3: 18.0,
                2: 3,
                5: 7,
                6: 10,  # weights define adjacency
            },
        )

        if use_rule:

            def rule_mysos(m, b):
                return (
                    [m.y[a] for a in m.mysosindex[b]],
                    [m.mysosweights[a] for a in m.mysosindex[b]],
                )

            model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)

        else:
            model.mysos = pyo.SOSConstraint(
                model.B,
                var=model.y,
                sos=n,
                index=model.mysosindex,
                weights=model.mysosweights,
            )

    else:
        raise NotImplementedError

    return model


# *****************************************************************************
# *****************************************************************************

if __name__ == "__main__":
    unittest.main()

# *****************************************************************************
# *****************************************************************************
