# ******************************************************************************
# ******************************************************************************

import pyomo.environ as pyo
import math

# ******************************************************************************
# ******************************************************************************

# test sos1


def solve_problems():
    opt = pyo.SolverFactory("cbc")

    # opt = pyo.SolverFactory('scip')

    # opt = pyo.SolverFactory('cplex')

    # opt = pyo.SolverFactory('gurobi')

    show_output = False

    # **************************************************************************

    for sos_type_minus_one in range(2):
        number_problems = 7

        for i in range(number_problems):
            try:
                model, exp_res, abs_tol = problem_nonindexed_sosn(
                    version=i, n=sos_type_minus_one + 1
                )
            except NotImplementedError:
                continue

            opt.solve(model, tee=show_output)

            assert len(model.mysos) != 0

            assert math.isclose(pyo.value(model.OBJ), exp_res, abs_tol=abs_tol)

    # **************************************************************************

    # indexed SOS constraints

    # **************************************************************************

    for sos_type_minus_one in range(2):
        # indexed sos1 constraints, same variables

        number_problems = 4

        for i in range(number_problems):
            try:
                model, exp_res, abs_tol = problem_indexed_sosn(
                    version=i, n=sos_type_minus_one + 1
                )
            except NotImplementedError:
                continue

            problem = model.create_instance()

            opt.solve(problem, tee=show_output)

            assert len(problem.mysos) != 0

            # problem.pprint()
            # problem.mysos.pprint()

            assert math.isclose(pyo.value(problem.OBJ), exp_res, abs_tol=abs_tol)

    # **************************************************************************


# ******************************************************************************
# ******************************************************************************


def problem_nonindexed_sosn(version: int == 0, n: int = 1):
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

    if version == 0:
        # index is not provided, nor are the weights
        model.mysos = pyo.SOSConstraint(
            var=model.y, sos=n  # the SOS1 is made up of y[1], y[2] and y[4]
        )

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 1:
        # no weights, but use a list

        index = [2, 4, 6]  # the SOS2 is made up of y[2], y[4] and y[6]

        model.mysos = pyo.SOSConstraint(var=model.y, index=index, sos=n)

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 2:
        # no weights, but use pyo.Set component (has to be part of the model)

        model.mysosindex = pyo.Set(
            initialize=[2, 4, 6],  # the SOS2 is made up of y[2], y[4] and y[6]
            within=model.A,
        )

        model.mysos = pyo.SOSConstraint(var=model.y, index=model.mysosindex, sos=n)

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 3:
        # with weights, using a list and a dict

        index = [2, 4, 6]  # the SOS2 is made up of y[2] and y[4]
        weights = {2: 25.0, 4: 18.0, 6: 22}  # these are the respective weights

        model.mysos = pyo.SOSConstraint(
            var=model.y, index=index, weights=weights, sos=n
        )

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 4:
        # with weights, using a set and a param

        model.mysosindex = pyo.Set(
            initialize=[2, 4, 6],  # the SOS2 is made up of y[2], y[4] and y[6]
            within=model.A,
        )
        model.mysosweights = pyo.Param(
            model.mysosindex,
            initialize={2: 25.0, 4: 18.0, 6: 22},  # these are the respective weights
        )

        model.mysos = pyo.SOSConstraint(
            var=model.y, index=model.mysosindex, weights=model.mysosweights, sos=n
        )

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 5:
        # index is not provided, but the weights are provided as a dict

        weights = {1: 3, 2: 25.0, 4: 18.0, 6: 22}

        model.mysos = pyo.SOSConstraint(
            var=model.y,  # the SOS1 is made up of y[1], y[2] and y[4]
            sos=n,
            weights=weights,  # these are the respective weights
        )

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 6:
        # index is not provided, but the weights are provided as a dict

        model.mysosweights = pyo.Param(
            [1, 2, 4, 6],
            initialize={1: 3, 2: 25.0, 4: 18.0, 6: 22},  # these are the weights
        )

        model.mysos = pyo.SOSConstraint(
            var=model.y,  # the SOS1 is made up of y[1], y[2] and y[4]
            sos=n,
            weights=model.mysosweights,
        )

        if n == 1:
            exp_res = 0.04999999999999999
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    else:
        raise NotImplementedError

    return model, exp_res, abs_tol


# ******************************************************************************
# ******************************************************************************


def problem_indexed_sosn(version: int == 0, n: int = 1):
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

    if version == 0:
        # with index, no weights, using a dict

        index = {2: [1, 3], 4: [2, 5, 6]}

        model.mysos = pyo.SOSConstraint(model.B, var=model.y, sos=n, index=index)

        if n == 1:
            exp_res = -7.5000000000e-02
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 1:
        # with index, no weights, using a Set object

        model.mysosindex = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})

        model.mysos = pyo.SOSConstraint(
            model.B, var=model.y, sos=n, index=model.mysosindex
        )

        if n == 1:
            exp_res = -7.5000000000e-02
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 2:
        # with weights, provided using a set and a dict

        index = {2: [1, 3], 4: [2, 5, 6]}
        weights = {1: 25.0, 3: 18.0, 2: 35, 5: 7, 6: 10}

        model.mysos = pyo.SOSConstraint(
            model.B, var=model.y, sos=n, index=index, weights=weights
        )

        if n == 1:
            exp_res = -7.5000000000e-02
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    elif version == 3:
        # with weights, using a Set and a Param

        model.mysosindex = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})
        model.mysosweights = pyo.Param(
            model.A,  # model.A or a subset that covers all relevant members
            initialize={1: 25.0, 3: 18.0, 2: 35, 5: 7, 6: 10},
        )

        model.mysos = pyo.SOSConstraint(
            model.B,
            var=model.y,
            sos=n,
            index=model.mysosindex,
            weights=model.mysosweights,
        )

        if n == 1:
            exp_res = -7.5000000000e-02
            abs_tol = 1e-3
        else:
            exp_res = -0.07500000000000001
            abs_tol = 1e-3

    else:
        raise NotImplementedError

    return model, exp_res, abs_tol


# ******************************************************************************
# ******************************************************************************

solve_problems()

# ******************************************************************************
# ******************************************************************************
