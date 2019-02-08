import pyomo.environ as aml


def create_model1():
    m = aml.ConcreteModel()
    m._name = 'model1'
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    # m.d = aml.Constraint(expr=aml.inequality(-18, m.x[2] ** 2 + m.x[1],  28))
    m.o = aml.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model2():
    m = aml.ConcreteModel()
    m._name = 'model2'
    m.x = aml.Var([1, 2], initialize=4.0)
    m.d = aml.Constraint(expr=m.x[1] + m.x[2] <= 5)
    m.o = aml.Objective(expr=m.x[1] ** 2 + 4 * m.x[2] ** 2 - 8 * m.x[1] - 16 * m.x[2])
    m.x[1].setub(3.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model3():
    model = aml.ConcreteModel()
    model._name = 'model3'
    model.n_vars = 10
    model.lb = -10
    model.ub = 20
    model.init = 2.0
    model.index_vars = range(model.n_vars)
    model.x = aml.Var(model.index_vars, initialize=model.init, bounds=(model.lb, model.ub))

    def rule_constraint(m, i):
        return m.x[i] ** 3 - 1.0 == 0

    model.c = aml.Constraint(model.index_vars, rule=rule_constraint)

    model.obj = aml.Objective(expr=sum((model.x[i] - 2) ** 2 for i in model.index_vars))

    return model


def create_model4():

    m = aml.ConcreteModel()
    m._name = 'model4'
    m.x = aml.Var([1, 2], initialize=1.0)
    m.c1 = aml.Constraint(expr=m.x[1] + m.x[2] - 1 == 0)
    m.obj = aml.Objective(expr=2 * m.x[1] ** 2 + m.x[2] ** 2)
    return m


def create_model5():
    model = aml.ConcreteModel()
    model._name = 'model5'
    N = 100
    model.x = aml.Var(aml.RangeSet(1, N), initialize=1.0 / N)

    def f_rule(model):
        return 0
    model.f = aml.Objective(rule=f_rule)

    def cons1_rule(model, i):
        return i * (aml.cos(model.x[i]) + aml.sin(model.x[i])) + sum(aml.cos(model.x[j]) for j in range(1, N + 1)) - (N + i) == 0
    model.cons = aml.Constraint(aml.RangeSet(1, N), rule=cons1_rule)

    return model


def create_model6():
    model = aml.ConcreteModel()
    model._name = 'model6'
    model.S = [1, 2]
    model.x = aml.Var(model.S, initialize=1.0)

    def f(model):
        return model.x[1] ** 4 + (model.x[1] + model.x[2]) ** 2 + (-1.0 + aml.exp(model.x[2])) ** 2

    model.f = aml.Objective(rule=f)
    return model


def create_model7():
    model = aml.ConcreteModel()
    model._name = 'model7'
    M = 1000
    N = 3 * M
    model.M = aml.Param(initialize=M)
    model.N = aml.Param(initialize=3 * M)

    model.alpha = aml.Param(initialize=1.0)
    model.beta = aml.Param(initialize=0.0)
    model.gamma = aml.Param(initialize=0.125)
    model.delta = aml.Param(initialize=0.125)

    model.S1 = [i for i in range(1, 5)]
    model.S2 = [i for i in range(1, N + 1)]
    model.K = aml.Param(model.S1, initialize=0)
    model.x = aml.Var(model.S2, initialize=2.0)

    model.S3 = [i for i in range(1, N)]
    model.S4 = [i for i in range(1, 2 * M + 1)]
    model.S5 = [i for i in range(1, M + 1)]

    def f(model):
        exp1 = sum([aml.value(model.alpha) * model.x[i] ** 2 * (i / aml.value(model.N)) ** aml.value(model.K[1]) for i in model.S2])
        exp2 = sum([aml.value(model.beta) * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (
        i / aml.value(model.N)) ** aml.value(model.K[2]) for i in model.S3])
        exp3 = sum([aml.value(model.gamma) * model.x[i] ** 2 * model.x[i + aml.value(model.M)] ** 4 * (
        i / aml.value(model.N)) ** aml.value(model.K[3]) for i in model.S4])
        exp4 = sum([aml.value(model.delta) * model.x[i] * model.x[i + 2 * aml.value(model.M)] * (i / aml.value(model.N)) ** aml.value(
            model.K[4]) for i in model.S5])
        return 1.0 + exp1 + exp2 + exp3 + exp4

    model.f = aml.Objective(rule=f)

    return model


def create_model8():

    # cliff cute
    model = aml.ConcreteModel()
    model._name = 'cliff'
    model.x = aml.Var([1, 2], initialize=0.0)
    model.x[2].value = -1.0

    def f(model):
        return (0.01 * model.x[1] - 0.03) ** 2 - model.x[1] + model.x[2] + aml.exp(20 * (model.x[1] - model.x[2]))

    model.f = aml.Objective(rule=f)
    return model


def create_model9():

    # clplatea OXR2-MN-V-0
    model = aml.ConcreteModel()
    model._name = 'clplatea'

    p = 71
    wght = -0.1
    hp2 = 0.5 * p ** 2

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + \
                   0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + \
                   hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4 + \
                   hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4 \
                   for i in range(2, p + 1) for j in range(2, p + 1)) + (wght * model.x[p, p])

    model.f = aml.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model


def create_model10():

    # chainwoo SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'chainwoo'

    ns = 499.0
    n = 2 * ns + 2.0

    model.N = aml.RangeSet(1, n)

    def x_init_rule(model, i):
        if i == 1:
            return -3.0
        elif i == 2:
            return -1.0
        elif i == 3:
            return -3.0
        elif i == 4:
            return -1.0
        else:
            return -2.0

    model.x = aml.Var(model.N, initialize=x_init_rule)

    def f(model):
        return 1.0 + sum(100 * (model.x[2 * i] - model.x[2 * i - 1] ** 2) ** 2 + \
                         (1.0 - model.x[2 * i - 1]) ** 2 + 90 * (model.x[2 * i + 2] - model.x[2 * i + 1] ** 2) ** 2 + \
                         (1.0 - model.x[2 * i + 1]) ** 2 + \
                         10 * (model.x[2 * i] + model.x[2 * i + 2] - 2.0) ** 2 + \
                         (model.x[2 * i] - model.x[2 * i + 2]) ** 2 / 10 for i in range(1, int(ns) + 1))

    model.f = aml.Objective(rule=f)
    return model


def create_model11():
    model = aml.ConcreteModel()
    model._name = 'model11'
    n = 100
    c = 1.0

    def x_init(model, i):
        return i / float(n)

    model.x = aml.Param(aml.RangeSet(1, n), initialize=x_init)

    def w_init(model, i):
        return 1.0 / float(n)

    model.w = aml.Param(aml.RangeSet(1, n), initialize=w_init)

    model.h = aml.Var(aml.RangeSet(1, n), initialize=1.0, bounds=(0, None))

    model.f = aml.Objective(expr=0.0)

    def con1(model, i):
        return sum(-0.5 * c * model.w[j] * model.x[i] / (model.x[i] + model.x[j]) * model.h[i] * model.h[j] for j in
                   range(1, n + 1)) + model.h[i] == 1.0

    model.cons = aml.Constraint(aml.RangeSet(1, n), rule=con1)
    return model


def create_model12():

    # chenhark
    model = aml.ConcreteModel()
    model._name = 'chenhark'
    n = 1000
    nfree = 500
    ndegen = 200

    def x_p_init(model, i):
        if i <= 0:
            return 0.0
        elif i > nfree:
            return 0.0
        else:
            return 1.0

    model.x_p = aml.Param(aml.RangeSet(-1, n + 2), initialize=x_p_init)

    def x_init(model, i):
        return 0.5

    model.x = aml.Var(aml.RangeSet(1, n), initialize=x_init, bounds=(0.0, None))

    def f(model):
        return sum(0.5 * (model.x[i + 1] + model.x[i - 1] - 2 * model.x[i]) ** 2 for i in range(2, n)) + \
               0.5 * model.x[1] ** 2 + 0.5 * (2 * model.x[1] - model.x[2]) ** 2 + 0.5 * (2 * model.x[n] - model.x[
            n - 1]) ** 2 + \
               0.5 * (model.x[n]) ** 2 + sum(model.x[i] * (-6 * model.x_p[i] + \
                                                           4 * model.x_p[i + 1] + 4 * model.x_p[i - 1] - model.x_p[
                                                               i + 2] - model.x_p[i - 2]) for i in
                                             range(1, nfree + ndegen + 1)) + \
               sum(model.x[i] * (-6 * model.x_p[i] + 4 * model.x_p[i + 1] + 4 * model.x_p[i - 1] - \
                                 model.x_p[i + 2] - model.x_p[i - 2] + 1) for i in range(nfree + ndegen + 1, n + 1))

    model.f = aml.Objective(rule=f)
    return model


def create_model13():

    # aircrfta
    model = aml.ConcreteModel()
    model._name = 'aircrfta'

    model.rollrate = aml.Var(initialize=0.0)
    model.pitchrat = aml.Var(initialize=0.0)
    model.yawrate = aml.Var(initialize=0.0)
    model.attckang = aml.Var(initialize=0.0)
    model.sslipang = aml.Var(initialize=0.0)
    model.elevator = aml.Var(initialize=0.0)
    model.aileron = aml.Var(initialize=0.0)
    model.rudderdf = aml.Var(initialize=0.0)

    model.elevator = 0.1
    model.elevator.fixed = True
    model.aileron = 0
    model.aileron.fixed = True
    model.rudderdf = 0
    model.rudderdf.fixed = True

    def f(model):
        return 0

    model.f = aml.Objective(rule=f)

    def cons1(model):
        exp1 = -3.933 * model.rollrate + 0.107 * model.pitchrat + 0.126 * model.yawrate - 9.99 * model.sslipang \
               - 45.83 * model.aileron - 7.64 * model.rudderdf - 0.727 * model.pitchrat * model.yawrate + 8.39 * model.yawrate * model.attckang \
               - 684.4 * model.attckang * model.sslipang + 63.5 * model.pitchrat * model.attckang
        return (0, exp1)

    def cons2(model):
        exp2 = -0.987 * model.pitchrat - 22.95 * model.attckang - 28.37 * model.elevator + 0.949 * model.rollrate * model.yawrate \
               + 0.173 * model.rollrate * model.sslipang
        return (0, exp2)

    def cons3(model):
        exp3 = 0.002 * model.rollrate - 0.235 * model.yawrate + 5.67 * model.sslipang - 0.921 * model.aileron - 6.51 * model.rudderdf \
               - 0.716 * model.rollrate * model.pitchrat - 1.578 * model.rollrate * model.attckang + 1.132 * model.pitchrat * model.attckang
        return (0, exp3)

    def cons4(model):
        exp4 = model.pitchrat - model.attckang - 1.168 * model.elevator - model.rollrate * model.sslipang
        return (0, exp4)

    def cons5(model):
        exp5 = -model.yawrate - 0.196 * model.sslipang - 0.0071 * model.aileron + model.rollrate * model.attckang
        return (0, exp5)

    model.cons1 = aml.Constraint(rule=cons1)
    model.cons2 = aml.Constraint(rule=cons2)
    model.cons3 = aml.Constraint(rule=cons3)
    model.cons4 = aml.Constraint(rule=cons4)
    model.cons5 = aml.Constraint(rule=cons5)

    return model


def create_model14():

    # all_init
    model = aml.ConcreteModel()
    model._name = 'all_init'

    model.x = aml.Var(aml.RangeSet(1, 4))

    model.f = aml.Objective(expr=model.x[3] - 1 + model.x[1] ** 2 + model.x[2] ** 2 + (model.x[3] + model.x[4]) ** 2 + aml.sin(
        model.x[3]) ** 2 + model.x[1] ** 2 * model.x[2] ** 2 + model.x[4] - 3 + \
                             aml.sin(model.x[3]) ** 2 + (model.x[4] - 1) ** 2 + (model.x[2] ** 2) ** 2 + (
                                                                                                     model.x[3] ** 2 + (
                                                                                                     model.x[4] +
                                                                                                     model.x[
                                                                                                         1]) ** 2) ** 2 + \
                             (model.x[1] - 4 + aml.sin(model.x[4]) ** 2 + model.x[2] ** 2 * model.x[3] ** 2) ** 2 + aml.sin(
        model.x[4]) ** 4)

    model.cons1 = aml.Constraint(expr=model.x[2] >= 1)

    model.cons2 = aml.Constraint(expr=aml.inequality(-1e+10,model.x[3],1.0))

    model.cons3 = aml.Constraint(expr=model.x[4] == 2)

    return model


def create_model15():

    # block_qp5
    model = aml.ConcreteModel()
    model._name = 'block_qp5'

    model.n = aml.Param(initialize=1000)
    model.b = aml.Param(initialize=5)

    model.Sn = aml.RangeSet(1, model.n)
    model.Sb = aml.RangeSet(1, model.b)

    model.x = aml.Var(model.Sn, bounds=(-1, 1), initialize=0.99)
    model.y = aml.Var(model.Sn, bounds=(-1, 1), initialize=-0.99)
    model.z = aml.Var(model.Sb, bounds=(0, 1), initialize=0.5)

    def f(model):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in model.Sn:
            sum_expr_1 += (i / model.n) * model.x[i] * model.y[i]
        for j in model.Sb:
            sum_expr_2 += 0.5 * model.z[j] ** 2
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(model):
        sum_cexpr_1 = 0
        sum_cexpr_2 = 0
        for i in model.Sn:
            sum_cexpr_1 += model.x[i] + model.y[i]
        for j in model.Sb:
            sum_cexpr_2 += model.z[j]
        cexp = sum_cexpr_1 + sum_cexpr_2
        return (aml.value(model.b) + 1, cexp, None)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model, i):
        csum = 0
        for j in model.Sb:
            csum += model.z[j]
        return model.x[i] + model.y[i] + csum == aml.value(model.b)

    model.cons2 = aml.Constraint(model.Sn, rule=cons2)

    return model


def create_model16():

    # brownal
    model = aml.ConcreteModel()
    model._name = 'brownal'
    N = 10

    model.x = aml.Var(aml.RangeSet(1, N), initialize=0.5)

    def f_rule(model):
        expr = 1.0
        for j in range(1, N + 1):
            expr *= model.x[j]
        expr -= 1.0
        return sum(
            (model.x[i] + sum(model.x[j] for j in range(1, N + 1)) - (N + 1)) ** 2 for i in range(1, N)) + expr ** 2

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model17():

    #boydn3d
    model = aml.ConcreteModel()
    model._name = 'boydn3d'

    N = 10000
    kappa1 = 2.0
    kappa2 = 1.0

    model.x = aml.Var(aml.RangeSet(1, N), initialize=-1.0)

    def f_rule(model):
        return 0

    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return (-2 * model.x[2] + kappa2 + (3 - kappa1 * model.x[1]) * model.x[1]) == 0

    model.cons1 = aml.Constraint(rule=con1)

    def con2(model, i):
        return (-model.x[i - 1] - 2 * model.x[i + 1] + kappa2 + (3 - kappa1 * model.x[i]) * model.x[i]) == 0

    model.cons2 = aml.Constraint(aml.RangeSet(2, N - 1), rule=con2)

    def con3(model):
        return (-model.x[N - 1] + kappa2 + (3 - kappa1 * model.x[N]) * model.x[N]) == 0

    model.cons3 = aml.Constraint(rule=con3)

    return model


def create_model18():

    # bt10
    model = aml.ConcreteModel()
    model._name = 'bt10'

    model.S = aml.RangeSet(1, 2)
    model.x = aml.Var(model.S, initialize=2.0)

    def f(model):
        return -model.x[1]

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.x[2] - model.x[1] ** 3 == 0

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return -model.x[2] + model.x[1] ** 2 == 0

    model.cons2 = aml.Constraint(rule=cons2)

    return model


def create_model19():

    # bt11 OOR2-AY-5-3
    model = aml.ConcreteModel()
    model._name = 'bt11'

    model.x = aml.Var(aml.RangeSet(1, 5), initialize=2.0)

    def f(model):
        return (model.x[1] - 1.0) ** 2 + (model.x[1] - model.x[2]) ** 2 + \
               (model.x[2] - model.x[3]) ** 2 + (model.x[3] - model.x[4]) ** 4 + (model.x[4] - model.x[5]) ** 4

    model.f = aml.Objective(rule=f)

    def con1(model):
        return model.x[1] + model.x[2] ** 2 + model.x[3] ** 3 == -2 + (18.0) ** 0.5

    model.cons1 = aml.Constraint(rule=con1)

    def con2(model):
        return model.x[2] + model.x[4] - model.x[3] ** 2 == -2 + (8.0) ** 0.5

    model.cons2 = aml.Constraint(rule=con2)

    def con3(model):
        return model.x[1] - model.x[5] == 2.0

    model.cons3 = aml.Constraint(rule=con3)

    return model


def create_model20():

    # Aljazzaf QQR2-AN-3-1
    model = aml.ConcreteModel()
    model._name = 'Aljazzaf'

    model.N = aml.Param(initialize=3)
    model.N1 = aml.Param(initialize=2)
    model.Biga = aml.Param(initialize=100.0)

    def F_rule(model):
        return (aml.value(model.Biga) ** 2 - 1.0) / (aml.value(model.N) - 1)

    model.F = aml.Param(initialize=F_rule)

    def F2_rule(model):
        return (aml.value(model.Biga) ** 2 - 1.0) / (aml.value(model.Biga) * (aml.value(model.N) - 1))

    model.F2 = aml.Param(initialize=F2_rule)
    model.S = aml.RangeSet(1, model.N)

    def A_rule(model, i):
        return aml.value(model.Biga) - (i - 1) * aml.value(model.F2)

    model.A = aml.Param(model.S, initialize=A_rule)

    def B_rule(model, i):
        return (i - 1) * aml.value(model.F) + 1.0

    model.B = aml.Param(model.S, initialize=B_rule)

    model.x = aml.Var(model.S, bounds=(0, None), initialize=0.0)

    model.SS1 = aml.RangeSet(2, model.N1)
    model.SS2 = aml.RangeSet(model.N1 + 1, model.N)

    def f(model):
        return model.A[1] * (model.x[1] - 0.5) ** 2 + \
               sum(aml.value(model.A[i]) * (model.x[i] + 1.0) ** 2 for i in model.SS1) + \
               sum(aml.value(model.A[i]) * (model.x[i] - 1.0) ** 2 for i in model.SS2)

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -model.B[1] * model.x[1] + model.B[1] + \
               sum(aml.value(model.B[i]) * (model.x[i] - 0.0) ** 2 for i in model.SS1) + \
               sum(aml.value(model.B[i]) * (model.x[i] - 1.0) ** 2 for i in model.SS2) == 0

    model.cons1 = aml.Constraint(rule=cons1)
    return model


# ToDo: check with line search
def create_model21():

    # allinitc OOR2-AY-4-1
    model = aml.ConcreteModel()
    model._name = 'allinitc'

    model.x = aml.Var(aml.RangeSet(1, 4))
    model.f = aml.Objective(expr=model.x[3] - 1 + model.x[1] ** 2 + \
                             model.x[2] ** 2 + (model.x[3] + model.x[4]) ** 2 + \
                             aml.sin(model.x[3]) ** 2 + model.x[1] ** 2 * model.x[2] ** 2 + model.x[4] - 3 + \
                             aml.sin(model.x[3]) ** 2 + \
                             (model.x[4] - 1) ** 2 + \
                             (model.x[2] ** 2) ** 2 + \
                             (model.x[3] ** 2 + (model.x[4] + model.x[1]) ** 2) ** 2 + \
                             (model.x[1] - 4 + aml.sin(model.x[4]) ** 2 + model.x[2] ** 2 * model.x[3] ** 2) ** 2 + \
                             aml.sin(model.x[4]) ** 4)

    model.cons1 = aml.Constraint(expr=model.x[2] >= 1)
    model.cons2 = aml.Constraint(expr=aml.inequality(-1e10,model.x[3],1))
    model.cons3 = aml.Constraint(expr=model.x[4] == 2)
    model.cons4 = aml.Constraint(expr=model.x[1] ** 2 + model.x[2] ** 2 - 1 <= 0)
    return model


def create_model22():

    # allinitu OUR2-AY-4-0
    model = aml.ConcreteModel()
    model._name = 'allinitu'

    model.x = aml.Var(aml.RangeSet(1, 4))
    model.f = aml.Objective(expr=model.x[3] - 1 +
                             model.x[1] ** 2 + \
                             model.x[2] ** 2 + (model.x[3] + model.x[4]) ** 2 + \
                             aml.sin(model.x[3]) ** 2 + model.x[1] ** 2 * model.x[2] ** 2 + model.x[4] - 3 + \
                             aml.sin(model.x[3]) ** 2 + \
                             (model.x[4] - 1) ** 2 + \
                             (model.x[2] ** 2) ** 2 + \
                             (model.x[3] ** 2 + (model.x[4] + model.x[1]) ** 2) ** 2 + \
                             (model.x[1] - 4 + aml.sin(model.x[4]) ** 2 + model.x[2] ** 2 * model.x[3] ** 2) ** 2 + \
                             aml.sin(model.x[4]) ** 4)
    return model


def create_model23():

    # allsotame OOR2-AN-2-1
    model = aml.ConcreteModel()
    model._name = 'allsotame'

    model.x = aml.Var()
    model.y = aml.Var()

    model.f = aml.Objective(expr=aml.exp(model.x - 2 * model.y))

    model.cons1 = aml.Constraint(expr=aml.sin(-model.x + model.y - 1) == 0)
    model.cons2 = aml.Constraint(expr=aml.inequality(-2, model.x, 2))
    model.cons3 = aml.Constraint(expr=aml.inequality(-1.5, model.y, 1.5))

    return model


def create_model24():

    # arglina SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'arglina'

    N = 100.0
    M = 200.0

    model.x = aml.Var(aml.RangeSet(1, N), initialize=1.0)

    def f_rule(model):
        return sum((sum(-2.0 * model.x[j] / M for j in range(1, i)) + \
                    model.x[i] * (1.0 - 2.0 / M) + \
                    sum(-2.0 * model.x[j] / M for j in range(i + 1, int(N) + 1)) - \
                    1.0) ** 2 for i in range(1, int(N) + 1)) + \
               sum((sum(-2.0 * model.x[j] / M for j in range(1, int(N) + 1)) - 1.0) ** 2 \
                   for i in range(int(N) + 1, int(M) + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model25():

    # arglinb SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'arglinb'

    model.N = 10
    model.M = 20
    model.x = aml.Var(aml.RangeSet(1, model.N), initialize=1.0)

    def f_rule(model):
        return sum((sum(model.x[j] * i * j for j in range(1, model.N + 1)) - 1.0) ** 2 for i in range(1, model.M + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model26():

    # arglinac SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'arglinac'

    model.N = 10
    model.M = 20

    model.x = aml.Var(aml.RangeSet(1, model.N), initialize=1.0)

    # 2 + sum {i in 2..M-1} (sum {j in 2..N-1} x[j]*j*(i-1) - 1.0)^2

    def f_rule(model):
        # return sum((sum(model.x[j]*j*(i-1)for j in range(2,model.N)))-1.0)**2 for i in range(2,model.M))
        return 2 + sum((sum(model.x[j] * j * (i - 1) for j in range(2, model.N)) - 1.0) ** 2 for i in range(2, model.M))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model27():

    # argtrig NOR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'argtrig'

    N = 100
    model.x = aml.Var(aml.RangeSet(1, N), initialize=1.0 / N)

    def f_rule(model):
        return 0

    model.f = aml.Objective(rule=f_rule)

    def cons1_rule(model, i):
        return i * (aml.cos(model.x[i]) + aml.sin(model.x[i])) + sum(aml.cos(model.x[j]) for j in range(1, N + 1)) - (N + i) == 0

    model.cons = aml.Constraint(aml.RangeSet(1, N), rule=cons1_rule)
    return model


# ToDo: try with line search WORKS WITH LS
def create_model28():

    # artif NOR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'artif'

    N = 5000

    model.S = aml.RangeSet(0, N + 1)
    model.SS = aml.RangeSet(1, N)

    model.x = aml.Var(model.S, initialize=1.0)

    def f(model):
        return 0

    model.f = aml.Objective(rule=f)

    def cons(model, i):
        expr = (-0.05 * (model.x[i] + model.x[i + 1] + model.x[i - 1]) + aml.atan(aml.sin((i % 100) * model.x[i])))
        return (0, expr)

    model.cons = aml.Constraint(model.SS, rule=cons)

    model.x[0] = 0.0
    model.x[0].fixed = True

    model.x[N + 1] = 0.0
    model.x[N + 1].fixed = True
    return model


def create_model29():

    # arwhead OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'arwhead'

    model.N = aml.Param(initialize=5000)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, initialize=1.0)
    model.SS = aml.RangeSet(1, model.N - 1)

    def f(model):
        expsum1 = sum((-4 * model.x[i] + 3.0) for i in model.SS)
        expsum2 = sum((model.x[i] ** 2 + model.x[aml.value(model.N)] ** 2) ** 2 for i in model.SS)
        return expsum1 + expsum2

    model.f = aml.Objective(rule=f)
    return model


def create_model30():

    # aug2d QLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'aug2d'

    model.nx = 10
    model.ny = 10

    model.x = aml.Var(aml.RangeSet(1, model.nx), aml.RangeSet(0, model.ny + 1))
    model.y = aml.Var(aml.RangeSet(0, model.nx + 1), aml.RangeSet(1, model.ny))

    model.snx = aml.RangeSet(2, model.nx - 1)
    model.sny = aml.RangeSet(2, model.ny - 1)

    def f_rule(model):
        return (sum((model.x[i, j] - 1) ** 2 for i in range(1, model.nx) for j in range(1, model.ny)) + \
                sum((model.y[i, j] - 1) ** 2 for i in range(1, model.nx) for j in range(1, model.ny)) + \
                sum((model.x[i, model.ny] - 1) ** 2 for i in range(1, model.nx)) + \
                sum((model.y[model.nx, j] - 1) ** 2 for j in range(1, model.ny))) / 2.0

    model.f = aml.Objective(rule=f_rule)

    def v1(model, i, j):
        return ((model.x[i, j] - model.x[i - 1, j]) + (model.y[i, j] - model.y[i, j - 1]) - 1) == 0

    model.v1 = aml.Constraint(model.snx, model.sny, rule=v1)

    def v2(model, i):
        return (model.x[i, 0] + (model.x[i, 1] - model.x[i - 1, 1]) + model.y[i, 1] - 1) == 0

    model.v2 = aml.Constraint(model.snx, rule=v2)

    def v3(model, i):
        return model.x[i, model.ny + 1] + (model.x[i, model.ny] - model.x[i - 1, model.ny]) - model.y[
            i, model.ny - 1] - 1 == 0

    model.v3 = aml.Constraint(model.snx, rule=v3)

    def v4(model, j):
        return model.y[0, j] + (model.y[1, j] - model.y[1, j - 1]) + model.x[1, j] - 1 == 0

    model.v4 = aml.Constraint(model.sny, rule=v4)

    def v5(model, j):
        return model.y[model.nx + 1, j] + (model.y[model.nx, j] - model.y[model.nx, j - 1]) - model.x[
            model.nx - 1, j] - 1 == 0

    model.v5 = aml.Constraint(model.sny, rule=v5)

    return model


def create_model31():

    # avion2 OLR2-RN-49-15
    model = aml.ConcreteModel()
    model._name = 'avion2'

    model.SR = aml.Var(initialize=27.452,
                       bounds=(10, 150))

    model.LR = aml.Var(initialize=1.5000,
                       bounds=(0, 10))

    model.PK = aml.Var(initialize=10.000,
                       bounds=(0, 10))

    model.EF = aml.Var(initialize=0.000,
                       bounds=(0, 5))

    model.SX = aml.Var(initialize=19.217,
                       bounds=(7, 120))

    model.LX = aml.Var(initialize=1.5000,
                       bounds=(1.5, 8))

    model.SD = aml.Var(initialize=3.5688,
                       bounds=(2, 20))

    model.SK = aml.Var(initialize=4.0696,
                       bounds=(2, 30))

    model.ST = aml.Var(initialize=34.315,
                       bounds=(30, 500))

    model.SF = aml.Var(initialize=88.025,
                       bounds=(20, 200))

    model.LF = aml.Var(initialize=5.1306,
                       bounds=(0.01, 20))

    model.AM = aml.Var(initialize=0.0000,
                       bounds=(0, 10))

    model.CA = aml.Var(initialize=-0.14809,
                       bounds=(-0.2, -0.001))

    model.CB = aml.Var(initialize=0.75980,
                       bounds=(0.1, 2))

    model.SO = aml.Var(initialize=0.0000,
                       bounds=(0, 1))

    model.SS = aml.Var(initialize=0.0000,
                       bounds=(0, 2))

    model.IMPDER = aml.Var(initialize=114.7,
                           bounds=(100, 1000))

    model.IMPK = aml.Var(initialize=500.00,
                         bounds=(500, 5000))

    model.IMPFUS = aml.Var(initialize=1760.5,
                           bounds=(500, 5000))

    model.QI = aml.Var(initialize=2325.6,
                       bounds=(1000, 20000))

    model.PT = aml.Var(initialize=5.6788,
                       bounds=(2, 30))

    model.MV = aml.Var(initialize=14197.0,
                       bounds=(2000, 20000))

    model.MC = aml.Var(initialize=12589.0,
                       bounds=(3000, 30000))

    model.MD = aml.Var(initialize=28394.0,
                       bounds=(5000, 50000))

    model.PD = aml.Var(initialize=0.2000,
                       bounds=(0.2, 0.8))

    model.NS = aml.Var(initialize=1.0000,
                       bounds=(1, 5))

    model.VS = aml.Var(initialize=0.0000,
                       bounds=(0, 20))

    model.CR = aml.Var(initialize=100.00,
                       bounds=(100, 400))

    model.PM = aml.Var(initialize=15.000,
                       bounds=(4, 15))

    model.DV = aml.Var(initialize=0.0000,
                       bounds=(0, 10))

    model.MZ = aml.Var(initialize=500.00,
                       bounds=(500, 10000))

    model.VN = aml.Var(initialize=10.000,
                       bounds=(10, 50))

    model.QV = aml.Var(initialize=814.90,
                       bounds=(250, 5000))

    model.QF = aml.Var(initialize=3140.5,
                       bounds=(750, 15000))

    model.IMPTRAIN = aml.Var(initialize=1945.0,
                             bounds=(250, 3000))

    model.IMPMOT = aml.Var(initialize=190.85,
                           bounds=(10, 5000))

    model.IMPNMOT = aml.Var(initialize=35.000,
                            bounds=(35, 70))

    model.IMPPET = aml.Var(initialize=100.00,
                           bounds=(100, 3000))

    model.IMPPIL = aml.Var(initialize=200.00,
                           bounds=(200, 400))

    model.IMPCAN = aml.Var(initialize=120.00,
                           bounds=(120, 240))

    model.IMPSNA = aml.Var(initialize=700.00,
                           bounds=(700, 1900))

    model.MS = aml.Var(initialize=1000.0,
                       bounds=(100, 1000))

    model.EL = aml.Var(initialize=4.9367,
                       bounds=(2, 20))

    model.DE = aml.Var(initialize=0.0000,
                       bounds=(0, 1))

    model.DS = aml.Var(initialize=0.0000,
                       bounds=(0, 2))

    model.IMPVOIL = aml.Var(initialize=5000.0,
                            bounds=(500, 5000))

    model.NM = aml.Var(initialize=1.0,
                       bounds=(1, 2))

    model.NP = aml.Var(initialize=1.0,
                       bounds=(1, 2))

    model.NG = aml.Var(initialize=1.0,
                       bounds=(1, 2))

    def f(model):
        expr = (model.SK - 0.01 * model.PK * model.SR) ** 2 \
               + (model.CA - (model.SS - model.SO - model.CB * model.LF) / (model.LF ** 2)) ** 2 \
               + (-2 * model.AM + model.SO + model.SS + 0.01 * model.EF / model.LF) ** 2 \
               + (model.AM - 0.025 * model.SO * model.CB ** 2 / model.CA) ** 2 \
               + (model.IMPDER - 27.5 * model.SD - 1.3 * model.SD ** 2) ** 2 \
               + (model.IMPK - 70 * model.SK + 8.6 * model.SK ** 2) ** 2 \
               + (model.QI - 1000 + model.MV ** 2 / 24000) ** 2 \
               + (1000 * model.PT - model.MD * model.PD) ** 2 \
               + (model.VN + model.VS + model.QF / 790 + 2 - model.MZ / model.CR + model.DV * model.PT) ** 2 \
               + (model.IMPMOT - 1000 * model.PT / (model.PM + 20) - 12 * aml.sqrt(model.PT)) ** 2 \
               + (model.ST - 1.25 * model.SR * model.NM) ** 2 \
               + (model.SR - model.MD / model.MS) ** 2 \
               + (model.QV - 2.4 * model.SX * aml.sqrt(model.SX) * model.EL / aml.sqrt(model.LX)) ** 2 \
               + (model.SO - 0.785 * model.DE ** 2 * model.PT) ** 2 \
               + (model.SS - 0.785 * model.DS ** 2 * model.PT) ** 2 \
               + (model.CB - 2 * (model.VN - model.CA * model.LF ** 3) / (
        model.LF ** 2 * (3 - model.SO * model.LF))) ** 2 \
               + (model.IMPVOIL - 1.15 * model.SX * (15 + 0.15 * model.SX) * (
        8 + (model.MC * model.LX / (50 * model.SR * model.EL)) ** 1.5)) ** 2
        return expr

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return (0, model.SD - 0.13 * model.SR)

    #       return model.SD-0.13*model.SR == 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return (0, model.SX - 0.7 * model.SR)

    #       return model.SX-0.7*model.SR == 0
    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return (0, model.LX - model.LR)

    #       return model.LX-model.LR == 0
    model.cons3 = aml.Constraint(rule=cons3)

    def cons5(model):
        return model.SF - model.ST - 2 * model.SD - 2 * model.SX - 2 * model.SK == 0

    model.cons5 = aml.Constraint(rule=cons5)

    def cons11(model):
        return model.IMPFUS - 20 * model.SF == 0

    model.cons11 = aml.Constraint(rule=cons11)

    def cons12(model):
        return model.MD - 2 * model.MV == 0

    model.cons12 = aml.Constraint(rule=cons12)

    def cons15(model):
        return model.QF - model.QI - model.QV == 0

    model.cons15 = aml.Constraint(rule=cons15)

    def cons17(model):
        return model.IMPTRAIN - 0.137 * model.MV == 0

    model.cons17 = aml.Constraint(rule=cons17)

    def cons19(model):
        return model.IMPNMOT - 35 * model.NM == 0

    model.cons19 = aml.Constraint(rule=cons19)

    def cons20(model):
        return model.IMPPET - 0.043 * model.QI == 0

    model.cons20 = aml.Constraint(rule=cons20)

    def cons21(model):
        return model.IMPPIL - 200 * model.NP == 0

    model.cons21 = aml.Constraint(rule=cons21)

    def cons22(model):
        return model.IMPCAN - 120 * model.NG == 0

    model.cons22 = aml.Constraint(rule=cons22)

    def cons23(model):
        return model.IMPSNA - 300 * model.NS - 400 == 0

    model.cons23 = aml.Constraint(rule=cons23)

    def cons24(model):
        return model.MC - model.MV + 95 * model.NP + 70 * model.NG + 660 * model.NM + 0.5 * model.QI - 380 == 0

    model.cons24 = aml.Constraint(rule=cons24)

    def cons25(model):
        return model.MZ - model.IMPTRAIN + model.IMPNMOT + model.IMPPET + model.IMPPIL + model.IMPCAN + model.IMPSNA + 290 == 0

    model.cons25 = aml.Constraint(rule=cons25)

    return model


def create_model32():

    # bdexp OBR2-AY-V-0
    model = aml.ConcreteModel()
    model._name = 'bdexp'

    model.N = 5000
    model.ngs = 4998

    model.x = aml.Var(aml.RangeSet(1, model.N), initialize=1.0)

    def f_rule(model):
        return sum((model.x[i] + model.x[i + 1]) * aml.exp((model.x[i] + model.x[i + 1]) * (-model.x[i + 2])) for i in
                   range(1, model.ngs + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model33():

    # bdqrtic SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'bdqrtic'

    model.N = aml.Param(initialize=1000)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, initialize=1.0)

    model.SS = aml.RangeSet(1, model.N - 4)

    def f(model):
        expsum1 = sum([(-4 * model.x[i] + 3.0) ** 2 for i in model.SS])
        expsum2 = sum([(model.x[i] ** 2 + 2 * model.x[i + 1] ** 2 + 3 * model.x[i + 2] ** 2 + 4 * model.x[
            i + 3] ** 2 + 5 * model.x[aml.value(model.N)] ** 2) ** 2 for i in model.SS])
        return expsum1 + expsum2

    model.f = aml.Objective(rule=f)
    return model


def create_model34():

    # bdvalue NOR2-MN-V-V
    model = aml.ConcreteModel()
    model._name = 'bdvalue'

    model.ndp = 5002
    model.h = 1.0 / (model.ndp - 1)
    model.S = aml.RangeSet(1, model.ndp)
    model.SS = aml.RangeSet(2, model.ndp - 1)

    def x_init(model, i):
        return ((i - 1) * model.h) * ((i - 1) * model.h - 1)

    model.x = aml.Var(model.S, initialize=x_init)

    model.x[1] = 0
    model.x[1].fixed = True
    model.x[model.ndp].fixed = True

    def f(model):
        return 0

    model.f = aml.Objective(rule=f)

    def cons(model, i):
        return (-model.x[i - 1] + 2 * model.x[i] - model.x[i + 1] + 0.5 * model.h ** 2 * (
        model.x[i] + i * model.h + 1) ** 3) == 0

    model.cons = aml.Constraint(model.SS, rule=cons)

    return model


def create_model35():

    # beale SUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'beale'

    model.N = 2

    model.x = aml.Var(aml.RangeSet(1, model.N), initialize=1.0)

    def f(model):
        return (-1.5 + model.x[1] * (1.0 - model.x[2])) ** 2 + (-2.25 + model.x[1] * (1.0 - model.x[2] ** 2)) ** 2 + (
                                                                                                                     -2.625 +
                                                                                                                     model.x[
                                                                                                                         1] * (
                                                                                                                     1.0 -
                                                                                                                     model.x[
                                                                                                                         2] ** 3)) ** 2

    model.f = aml.Objective(rule=f)
    return model


def create_model36():

    # biggs3 SXR2-AN-6-0
    model = aml.ConcreteModel()
    model._name = 'biggs3'

    model.N = aml.Param(initialize=6)
    model.M = aml.Param(initialize=13)
    model.S = aml.RangeSet(1, model.N)
    model.SS = aml.RangeSet(1, model.M)

    model.x = aml.Var(model.S)
    model.x[1] = 1.0
    model.x[2] = 2.0
    model.x[3] = 1.0
    model.x[4] = 1.0
    model.x[5] = 4.0
    model.x[6] = 3.0

    def f_rule(model):
        sum1 = 0.0
        for i in model.SS:
            sum1 += (-aml.exp(-0.1 * i) + 5 * aml.exp(-i) - 3 * aml.exp(-0.4 * i) + model.x[3] * aml.exp(-0.1 * i * model.x[1]) \
                     - model.x[4] * aml.exp(-0.1 * i * model.x[2]) + model.x[6] * aml.exp(-0.1 * i * model.x[5])) ** 2
        return sum1

    model.f = aml.Objective(rule=f_rule)

    def cons1(model):
        return model.x[3] == 1

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return model.x[5] == 4

    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return model.x[6] == 3

    model.cons3 = aml.Constraint(rule=cons3)

    return model


def create_model37():

    # biggs5 SXR2-AN-6-0
    model = aml.ConcreteModel()
    model._name = 'biggs5'

    model.N = aml.Param(initialize=6)
    model.M = aml.Param(initialize=13)
    model.S = aml.RangeSet(1, model.N)
    model.SS = aml.RangeSet(1, model.M)
    model.xinit = dict()
    model.xinit[1] = 1
    model.xinit[2] = 2
    model.xinit[3] = 1
    model.xinit[4] = 1
    model.xinit[5] = 4
    model.xinit[6] = 3

    def init1(model, i):
        return model.xinit[i]

    model.x = aml.Var(model.S, initialize=init1)

    def f(model):
        sum1 = 0
        for i in model.SS:
            sum1 += (-aml.exp(-0.1 * i) + 5 * aml.exp(-i) - 3 * aml.exp(-0.4 * i) + model.x[3] * aml.exp(-0.1 * i * model.x[1]) \
                     - model.x[4] * aml.exp(-0.1 * i * model.x[2]) + model.x[6] * aml.exp(-0.1 * i * model.x[5])) ** 2
        return sum1

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.x[6] == 3

    model.cons1 = aml.Constraint(rule=cons1)
    return model


def create_model38():

    # biggs6 SUR2-AN-6-0
    model = aml.ConcreteModel()
    model._name = 'biggs6'

    model.N = 6
    model.M = 13
    model.s = aml.RangeSet(1, model.N)

    model.xinit = {}
    model.xinit[1] = 1
    model.xinit[2] = 2
    model.xinit[3] = 1
    model.xinit[4] = 1
    model.xinit[5] = 4
    model.xinit[6] = 3

    def x_init(model, i):
        return model.xinit[i]

    model.x = aml.Var(model.s, initialize=x_init)

    def f_rule(model):
        return sum((-aml.exp(-0.1 * i) + 5 * aml.exp(-i) - 3 * aml.exp(-0.4 * i) \
                    + model.x[3] * aml.exp(-0.1 * i * model.x[1]) - model.x[4] * aml.exp(-0.1 * i * model.x[2]) + \
                    model.x[6] * aml.exp(-0.1 * i * model.x[5])) ** 2 for i in range(1, model.M + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model39():

    # biggsb1 QBR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'biggsb1'

    model.N = 1000
    model.x = aml.Var(aml.RangeSet(1, model.N))

    def f_rule(model):
        return (model.x[1] - 1) ** 2 + sum((model.x[i + 1] - model.x[i]) ** 2 for i in range(1, model.N)) + (
                                                                                                            1 - model.x[
                                                                                                                model.N]) ** 2

    model.f = aml.Objective(rule=f_rule)

    def cons1(model, i):
        return aml.inequality(0.0, model.x[i], 0.9)

    model.cons1 = aml.Constraint(aml.RangeSet(1, model.N - 1), rule=cons1)

    return model


def create_model40():

    # biggsc41 QLR2-AN-4-7
    model = aml.ConcreteModel()
    model._name = 'biggsc41'

    model.x = aml.Var(aml.RangeSet(1, 4), bounds=(0, 5), initialize=0)

    def f_rule(model):
        return (-model.x[1] * model.x[3] - model.x[2] * model.x[4])

    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return aml.inequality(0.0, model.x[1] + model.x[2] - 2.5, 5.0)

    def con2(model):
        return aml.inequality(0, model.x[1] + model.x[3] - 2.5, 5.0)

    def con3(model):
        return aml.inequality(0, model.x[1] + model.x[4] - 2.5, 5.0)

    def con4(model):
        return aml.inequality(0, model.x[2] + model.x[3] - 2.0, 5.0)

    def con5(model):
        return aml.inequality(0, model.x[2] + model.x[4] - 2.0, 5.0)

    def con6(model):
        return aml.inequality(0,  model.x[3] + model.x[4] - 1.5, 5.0)

    def con7(model):
        return model.x[1] + model.x[2] + model.x[3] + model.x[4] - 5.0 >= 0

    model.cons1 = aml.Constraint(rule=con1)
    model.cons2 = aml.Constraint(rule=con2)
    model.cons3 = aml.Constraint(rule=con3)
    model.cons4 = aml.Constraint(rule=con4)
    model.cons5 = aml.Constraint(rule=con5)
    model.cons6 = aml.Constraint(rule=con6)
    model.cons7 = aml.Constraint(rule=con7)

    return model


def create_model41():

    # blockqp1 QLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'blockqp1'

    model.n = aml.Param(initialize=1000)
    model.b = aml.Param(initialize=5)

    model.Sn = aml.RangeSet(1, model.n)
    model.Sb = aml.RangeSet(1, model.b)

    model.x = aml.Var(model.Sn, bounds=(-1, 1), initialize=0.99)
    model.y = aml.Var(model.Sn, bounds=(-1, 1), initialize=-0.99)
    model.z = aml.Var(model.Sb, bounds=(0, 2), initialize=0.5)

    def f(model):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in model.Sn:
            sum_expr_1 += model.x[i] * model.y[i]
        for j in model.Sb:
            sum_expr_2 += 0.5 * model.z[j] ** 2
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(model):
        sum_cexpr_1 = 0
        sum_cexpr_2 = 0
        for i in model.Sn:
            sum_cexpr_1 += model.x[i] + model.y[i]
        for j in model.Sb:
            sum_cexpr_2 += model.z[j]
        cexp = sum_cexpr_1 + sum_cexpr_2
        return (aml.value(model.b) + 1, cexp, None)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model, i):
        csum = 0
        for j in model.Sb:
            csum += model.z[j]
        return model.x[i] + model.y[i] + csum == aml.value(model.b)

    model.cons2 = aml.Constraint(model.Sn, rule=cons2)
    return model


def create_model42():

    # blockqp2 QLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'blockqp2'

    model.n = aml.Param(initialize=1000)
    model.b = aml.Param(initialize=5)

    model.Sn = aml.RangeSet(1, model.n)
    model.Sb = aml.RangeSet(1, model.b)

    model.x = aml.Var(model.Sn, bounds=(-1, 1), initialize=0.99)
    model.y = aml.Var(model.Sn, bounds=(-1, 1), initialize=-0.99)
    model.z = aml.Var(model.Sb, bounds=(0, 2), initialize=0.5)

    def f(model):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in model.Sn:
            sum_expr_1 += model.x[i] * model.y[i]
        for j in model.Sb:
            sum_expr_2 += 0.5 * model.z[j] ** 2
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(model):
        sum_cexpr_1 = 0
        sum_cexpr_2 = 0
        for i in model.Sn:
            sum_cexpr_1 += model.x[i] + model.y[i]
        for j in model.Sb:
            sum_cexpr_2 += model.z[j]
        cexp = sum_cexpr_1 + sum_cexpr_2
        return (aml.value(model.b) + 1, cexp, None)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model, i):
        csum = 0
        for j in model.Sb:
            csum += model.z[j]
        return model.x[i] - model.y[i] + csum == aml.value(model.b)

    model.cons2 = aml.Constraint(model.Sn, rule=cons2)

    return model


# ToDo: try with line search
def create_model43():

    # blockqp3 QLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'blockqp3'

    model.n = aml.Param(initialize=1000)
    model.b = aml.Param(initialize=5)

    model.Sn = aml.RangeSet(1, model.n)
    model.Sb = aml.RangeSet(1, model.b)

    model.x = aml.Var(model.Sn, bounds=(-1, 1), initialize=0.99)
    model.y = aml.Var(model.Sn, bounds=(-1, 1), initialize=-0.99)
    model.z = aml.Var(model.Sb, bounds=(0, 2), initialize=0.5)

    def f(model):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in model.Sn:
            sum_expr_1 += (i / model.n) * model.x[i] * model.y[i]
        for j in model.Sb:
            sum_expr_2 += 0.5 * model.z[j] ** 2
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(model):
        sum_cexpr_1 = 0
        sum_cexpr_2 = 0
        for i in model.Sn:
            sum_cexpr_1 += model.x[i] + model.y[i]
        for j in model.Sb:
            sum_cexpr_2 += model.z[j]
        cexp = sum_cexpr_1 + sum_cexpr_2
        return (aml.value(model.b) + 1, cexp, None)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model, i):
        csum = 0
        for j in model.Sb:
            csum += model.z[j]
        return model.x[i] + model.y[i] + csum == aml.value(model.b)

    model.cons2 = aml.Constraint(model.Sn, rule=cons2)

    return model


# ToDo: try with line search
def create_model44():

    # blockqp4 QLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'blockqp4'

    model.n = aml.Param(initialize=1000)
    model.b = aml.Param(initialize=5)

    model.Sn = aml.RangeSet(1, model.n)
    model.Sb = aml.RangeSet(1, model.b)

    model.x = aml.Var(model.Sn, bounds=(-1, 1), initialize=0.99)
    model.y = aml.Var(model.Sn, bounds=(-1, 1), initialize=-0.99)
    model.z = aml.Var(model.Sb, bounds=(0, 2), initialize=0.5)

    def f(model):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in model.Sn:
            sum_expr_1 += (i / model.n) * model.x[i] * model.y[i]
        for j in model.Sb:
            sum_expr_2 += 0.5 * model.z[j] ** 2
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(model):
        sum_cexpr_1 = 0
        sum_cexpr_2 = 0
        for i in model.Sn:
            sum_cexpr_1 += model.x[i] + model.y[i]
        for j in model.Sb:
            sum_cexpr_2 += model.z[j]
        cexp = sum_cexpr_1 + sum_cexpr_2
        return (aml.value(model.b) + 1, cexp, None)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model, i):
        csum = 0
        for j in model.Sb:
            csum += model.z[j]
        return model.x[i] - model.y[i] + csum == aml.value(model.b)

    model.cons2 = aml.Constraint(model.Sn, rule=cons2)

    return model


# ToDo: try with line search WORK WITH LS
def create_model45():

    # bloweya QLR2-MN-V-V
    model = aml.ConcreteModel()
    model._name = 'bloweya'

    N = 1000.0
    A = 1 / 5.0
    B = 2 / 5.0
    C = 3 / 5.0
    D = 4 / 5.0
    INT = N * (1 + A + B - C - D)

    def v_rule(model, i):
        if (0 <= i) and (i <= N * A):
            return 1.0
        elif (N * A + 1 <= i) and (i <= N * B):
            return 1 - (i - N * A) * 2 / (N * (B - A))
        elif (N * B + 1 <= i) and (i <= N * C):
            return -1
        elif (N * C + 1 <= i) and (i <= N * D):
            return (-1 + (i - N * C) * 2 / (N * (D - C)))
        else:
            return 1.0

    model.v = aml.Param(aml.RangeSet(0, N), initialize=v_rule)

    def u_rule(model, i):
        return model.v[i]

    model.u = aml.Var(aml.RangeSet(0, N), bounds=(-1.0, 1.0), initialize=u_rule)
    model.w = aml.Var(aml.RangeSet(0, N), initialize=0.0)

    def f_rule(model):
        return -2 * model.u[0] * model.u[1] + model.u[0] ** 2 + \
               sum(-2 * model.u[i] * model.u[i + 1] + 2 * model.u[i] ** 2 for i in range(1, int(N))) + \
               model.u[N] ** 2 + sum(1 / N ** 2 * model.u[i] * model.w[i] for i in range(0, int(N) + 1)) + \
               sum(-1 / N ** 2 * model.v[i] * model.u[i] - 2 / N ** 2 * model.v[i] * model.w[i] for i in
                   range(0, int(N) + 1)) + \
               (model.v[1] - model.v[0]) * model.u[0] \
               + sum((model.v[i - 1] - 2 * model.v[i] + model.v[i + 1]) * model.u[i] for i in range(1, int(N))) + (
                                                                                                                  model.v[
                                                                                                                      N - 1] -
                                                                                                                  model.v[
                                                                                                                      N]) * \
                                                                                                                  model.u[
                                                                                                                      N]

    model.f = aml.Objective(rule=f_rule)

    def con1_rule(model):
        return 0.5 * model.u[0] + sum(model.u[i] for i in range(1, int(N))) + 0.5 * model.u[N] == 0.2 * INT

    model.cons1 = aml.Constraint(rule=con1_rule)

    def con2_rule(model):
        return model.u[0] - model.u[1] - 1 / N ** 2 * model.w[0] == 0

    model.cons2 = aml.Constraint(rule=con2_rule)

    def con3_rule(model, i):
        return 2 * model.u[i] - model.u[i + 1] - model.u[i - 1] - 1 / N ** 2 * model.w[i] == 0

    model.cons3 = aml.Constraint(aml.RangeSet(1, N - 1), rule=con3_rule)

    def con4_rule(model):
        return model.u[N] - model.u[N - 1] - 1 / N ** 2 * model.w[N] == 0

    model.cons4 = aml.Constraint(rule=con4_rule)

    return model


def create_model46():

    # booth NLR2-AN-2-2
    model = aml.ConcreteModel()
    model._name = 'booth'

    model.Sx = aml.RangeSet(1, 2)
    model.x = aml.Var(model.Sx)

    def f(model):
        return 0

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return (0, model.x[1] + 2 * model.x[2] - 7)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return (0, 2 * model.x[1] + model.x[2] - 5)

    model.cons2 = aml.Constraint(rule=cons2)
    return model


def create_model47():

    # box2 SXR2-AN-3-0
    model = aml.ConcreteModel()
    model._name = 'box2'

    model.M = 10

    model.xinit = {}
    model.xinit[1] = 0.0
    model.xinit[2] = 10.0
    model.xinit[3] = 1.0

    def xinit(model, i):
        return model.xinit[i]

    model.x = aml.Var(aml.RangeSet(1, 3), initialize=xinit)

    def t(model, i):
        return 0.1 * i

    model.t = aml.Param(aml.RangeSet(1, model.M), initialize=t)

    def f_rule(model):
        return sum((aml.exp(-model.t[i] * model.x[1]) - aml.exp(-model.t[i] * model.x[2]) - model.x[3] \
                    * aml.exp(-model.t[i]) + model.x[3] * aml.exp(-i)) ** 2 for i in range(1, model.M + 1))

    model.f = aml.Objective(rule=f_rule)

    def cons1(model):
        return model.x[3] == 1.0

    model.cons1 = aml.Constraint(rule=cons1)

    return model


def create_model48():

    # box3 SXR2-AN-3-0
    model = aml.ConcreteModel()
    model._name = 'box3'

    model.M = 10

    model.xinit = {}
    model.xinit[1] = 0.0
    model.xinit[2] = 10.0
    model.xinit[3] = 1.0

    def xinit(model, i):
        return model.xinit[i]

    model.x = aml.Var(aml.RangeSet(1, 3), initialize=xinit)

    def t(model, i):
        return 0.1 * i

    model.t = aml.Param(aml.RangeSet(1, model.M), initialize=t)

    def f_rule(model):
        return sum((aml.exp(-model.t[i] * model.x[1]) - aml.exp(-model.t[i] * model.x[2]) - model.x[3] \
                    * aml.exp(-model.t[i]) + model.x[3] * aml.exp(-i)) ** 2 for i in range(1, model.M + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model49():

    # bqp1var QBR2-AN-1-0
    model = aml.ConcreteModel()
    model._name = 'bqp1var'

    model.x1 = aml.Var(initialize=0.25)

    def f(model):
        return (model.x1 + model.x1 ** 2)

    model.f = aml.Objective(rule=f)

    model.cons1 = aml.Constraint(expr=aml.inequality(0.0, model.x1, 0.5))
    return model


# ToDo: (this one needs to step at tol 1e-7)
def create_model50():

    # bratu1d OXR2-MN-V-0
    model = aml.ConcreteModel()
    model._name = 'bratu1d'

    N = 1001
    L = -3.4
    h = 1.0 / (N + 1.0)

    def x(model, i):
        if (i == 0) or (i == N + 1):
            return 0.0
        else:
            return -0.1 * h * (i ** 2)

    model.x = aml.Var(aml.RangeSet(0, N + 1), initialize=x)

    def f_rule(model):
        return 2 * L * h * (aml.exp(model.x[1]) - aml.exp(model.x[0])) / (model.x[1] - model.x[0]) \
               + sum(2.0 * model.x[i] ** 2 / h for i in range(1, N + 1)) \
               - sum(2.0 * model.x[i] * model.x[i - 1] / h for i in range(1, N + 1)) \
               + sum(2.0 * L * h * (aml.exp(model.x[i + 1]) - aml.exp(model.x[i])) / (model.x[i + 1] - model.x[i]) for i in
                     range(1, N + 1))

    model.f = aml.Objective(rule=f_rule)

    model.x[0] = 0.0
    model.x[0].fixed = True
    model.x[N + 1] = 0.0
    model.x[N + 1].fixed = True
    return model


def create_model51():

    # brkmcc
    model = aml.ConcreteModel()
    model._name = 'brkmcc'

    model.x = aml.Var(aml.RangeSet(1, 2), initialize=2.0)

    def f_rule(model):
        return (model.x[1] - 2) ** 2 + (model.x[2] - 1) ** 2 + (1 / (1 - 0.25 * model.x[1] ** 2 - \
                                                                     model.x[2] ** 2)) / 25 + 5 * (model.x[1] - 2 *
                                                                                                   model.x[2] + 1) ** 2

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model52():

    # browbs SUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'browbs'

    N = 2

    model.x = aml.Var(aml.RangeSet(1, N), initialize=1.0)

    def f_rule(model):
        return sum((model.x[i] - 1000000) ** 2 for i in range(1, N)) + \
               sum((model.x[i + 1] - 0.000002) ** 2 for i in range(1, N)) + \
               sum((model.x[i] * model.x[i + 1] - 2.0) ** 2 for i in range(1, N))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model53():

    # broyden7d OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'broyden7d'

    N = 1000

    model.x = aml.Var(aml.RangeSet(1, N), initialize=1.0)

    def f_rule(model):
        return (abs(-2 * model.x[2] + 1 + (3 - 2 * model.x[1]) * model.x[1])) ** (7 / 3.0) + \
               sum((abs(1 - model.x[i - 1] - 2 * model.x[i + 1] + (3 - 2 * model.x[i]) * model.x[i])) ** (7 / 3.0) for i
                   in range(2, N)) + \
               (abs(-model.x[N - 1] + 1 + (3 - 2 * model.x[N]) * model.x[N])) ** (7 / 3.0) + \
               sum((abs(model.x[i] + model.x[i + N / 2])) ** (7 / 3.0) for i in range(1, int(N / 2) + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model54():

    # broydenbd SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'broydenbd'

    N = 5000
    ml = 5
    mu = 1

    model.x = aml.Var(aml.RangeSet(1, N), initialize=-1)

    def j_init(model, i):
        return [j for j in range(1, N + 1) if (j != i) and (max(1, i - ml) <= j) and (j <= min(N, i + mu))]

    model.J = aml.Set(aml.RangeSet(1, N), initialize=j_init)

    def f_rule(model):
        return sum((model.x[i] * (2 + 5 * model.x[i] ** 2) + 1 - \
                    sum(model.x[j] * (1 + model.x[j]) for j in model.J[i])) ** 2 for i in range(1, N + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model55():

    # brybnd SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'brybnd'

    N = 5000
    ml = 5
    mu = 1

    model.x = aml.Var(aml.RangeSet(1, N), initialize=-1)

    def j_init(model, i):
        return [j for j in range(1, N + 1) if (j != i) and (max(1, i - ml) <= j) and (j <= min(N, i + mu))]

    model.J = aml.Set(aml.RangeSet(1, N), initialize=j_init)

    def f_rule(model):
        return sum((model.x[i] * (2 + 5 * model.x[i] ** 2) + 1 - \
                    sum(model.x[j] * (1 + model.x[j]) for j in model.J[i])) ** 2 for i in range(1, N + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model56():

    # bt1 QQR2-AN-2-1
    model = aml.ConcreteModel()
    model._name = 'bt1'

    xinit = dict()
    xinit[1] = 0.08
    xinit[2] = 0.06

    model.x = aml.Var(aml.RangeSet(1, 2), initialize=xinit)


    def f_rule(model):
        return 100 * model.x[1] ** 2 + 100 * model.x[2] ** 2 - model.x[1] - 100

    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return model.x[1] ** 2 + model.x[2] ** 2 - 1.0 == 0

    model.cons1 = aml.Constraint(rule=con1)

    return model


def create_model57():

    # bt2 QQR2-AY-3-1
    model = aml.ConcreteModel()
    model._name = 'bt2'

    model.S = aml.RangeSet(1, 3)
    model.x = aml.Var(model.S, initialize=10.0)

    def f(model):
        return (model.x[1] - 1.0) ** 2 + (model.x[1] - model.x[2]) ** 2 + (model.x[2] - model.x[3]) ** 4

    model.f = aml.Objective(rule=f, sense=aml.minimize)

    def cons1(model):
        return model.x[1] * (1.0 + model.x[2] ** 2) + model.x[3] ** 4 == 8.2426407

    model.cons1 = aml.Constraint(rule=cons1)

    return model


def create_model58():

    # bt12 QQR2-AN-5-3
    model = aml.ConcreteModel()
    model._name = 'bt12'

    xinit = dict()
    xinit[1] = 15.81
    xinit[2] = 1.58
    xinit[3] = 0.0
    xinit[4] = 15.083
    xinit[5] = 3.7164

    model.x = aml.Var(aml.RangeSet(1, 5), initialize=xinit)

    def f_rule(model):
        return 0.01 * model.x[1] ** 2 + model.x[2] ** 2

    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return model.x[1] + model.x[2] - model.x[3] ** 2 == 25.0

    def con2(model):
        return model.x[1] ** 2 + model.x[2] ** 2 - model.x[4] ** 2 == 25.0

    def con3(model):
        return model.x[1] - model.x[5] ** 2 == 2.0

    model.cons1 = aml.Constraint(rule=con1)
    model.cons2 = aml.Constraint(rule=con2)
    model.cons3 = aml.Constraint(rule=con3)
    return model


def create_model59():

    # bt13 LQR2-AY-5-1
    model = aml.ConcreteModel()
    model._name = 'bt13'

    model.N = 5
    model.S = aml.RangeSet(1, model.N)
    x_init = dict()
    x_init[1] = 1.0
    x_init[2] = 2.0
    x_init[3] = 3.0
    x_init[4] = 3.0
    x_init[5] = 228.0

    model.x = aml.Var(model.S, initialize=x_init)

    def f(model):
        return model.x[5]
    model.f = aml.Objective(rule=f, sense=aml.minimize)

    def cons1(model):
        return model.x[1] ** 2 + (model.x[1] - 2 * model.x[2]) ** 2 + (model.x[2] - 3 * model.x[3]) ** 2 + (model.x[
                                                                                                                3] - 4 *
                                                                                                            model.x[
                                                                                                                4]) ** 2 - \
               model.x[5] ** 2 == 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return model.x[5] >= 0
    model.cons2 = aml.Constraint(rule=cons2)

    return model


def create_model60():

    # bt3 SLR2-AY-5-3
    model = aml.ConcreteModel()
    model._name = 'bt3'

    model.S = aml.RangeSet(1, 5)
    model.x = aml.Var(model.S, initialize=20.0)

    def f(model):
        return (model.x[1] - model.x[2]) ** 2 + (model.x[2] + model.x[3] - 2) ** 2 + (model.x[4] - 1) ** 2 + (model.x[
                                                                                                                  5] - 1) ** 2
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.x[1] + 3 * model.x[2] == 0

    def cons2(model):
        return model.x[3] + model.x[4] - 2 * model.x[5] == 0

    def cons3(model):
        return model.x[2] - model.x[5] == 0

    model.cons1 = aml.Constraint(rule=cons1)
    model.cons2 = aml.Constraint(rule=cons2)
    model.cons3 = aml.Constraint(rule=cons3)

    return model


def create_model61():

    # bt4 QQR2-AN-3-2
    model = aml.ConcreteModel()
    model._name = 'bt4'

    xinit = dict()
    xinit[1] = 4.0382
    xinit[2] = -2.9470
    xinit[3] = -0.09115
    model.x = aml.Var(aml.RangeSet(1, 3), initialize=xinit)

    def f_rule(model):
        return model.x[1] - model.x[2] + model.x[2] ** 3
    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return -25 + model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 == 0
    model.cons1 = aml.Constraint(rule=con1)

    def con2(model):
        return model.x[1] + model.x[2] + model.x[3] - 1 == 0
    model.cons2 = aml.Constraint(rule=con2)

    return model


def create_model62():

    # bt5 QQR2-AN-3-2
    model = aml.ConcreteModel()
    model._name = 'bt5'

    model.N = 3
    x_init = dict()
    x_init[1] = 2.0
    x_init[2] = 2.0
    x_init[3] = 2.0

    model.x = aml.Var(aml.RangeSet(1, model.N), initialize=x_init)

    def f_rule(model):
        return 1000 - model.x[1] ** 2 - model.x[3] ** 2 - 2 * model.x[2] ** 2 - model.x[1] * model.x[2] - \
               model.x[1] * model.x[3]
    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return -25 + model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 == 0
    model.cons1 = aml.Constraint(rule=con1)

    def con2(model):
        return 8 * model.x[1] + 14 * model.x[2] + 7 * model.x[3] - 56 == 0
    model.cons2 = aml.Constraint(rule=con2)

    return model


def create_model63():

    # bt6 OOR2-AY-5-2
    model = aml.ConcreteModel()
    model._name = 'bt6'

    model.x = aml.Var(aml.RangeSet(1, 5), initialize=2.0)

    def f_rule(model):
        return (model.x[1] - 1.0) ** 2 + (model.x[1] - model.x[2]) ** 2 + (model.x[3] - 1.0) ** 2 + \
               (model.x[4] - 1.0) ** 4 + (model.x[5] - 1.0) ** 6
    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return model.x[4] * model.x[1] ** 2 + aml.sin(model.x[4] - model.x[5]) == 2 * (2.0) ** 0.5

    def con2(model):
        return model.x[3] ** 4 * model.x[2] ** 2 + model.x[2] == 8 + (2.0) ** 0.5

    model.cons1 = aml.Constraint(rule=con1)
    model.cons2 = aml.Constraint(rule=con2)
    return model


# ToDo: try with line search
def create_model64():

    # bt7 OQR2-AN-5-3
    model = aml.ConcreteModel()
    model._name = 'bt7'

    x_init = dict()
    x_init = dict()
    x_init[1] = -2.0
    x_init[2] = 1.0
    x_init[3] = 1.0
    x_init[4] = 1.0
    x_init[5] = 1.0

    model.x = aml.Var(aml.RangeSet(1, 5), initialize=x_init)

    def f_rule(model):
        return 100 * (model.x[2] - model.x[1] ** 2) ** 2 + (model.x[1] - 1.0) ** 2
    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return model.x[1] * model.x[2] - model.x[3] ** 2 == 1.0

    def con2(model):
        return model.x[2] ** 2 - model.x[4] ** 2 + model.x[1] == 0.0

    def con3(model):
        return model.x[5] ** 2 + model.x[1] == 0.5

    model.cons1 = aml.Constraint(rule=con1)
    model.cons2 = aml.Constraint(rule=con2)
    model.cons3 = aml.Constraint(rule=con3)

    return model


def create_model65():

    # bt8 QQR2-AN-5-2
    model = aml.ConcreteModel()
    model._name = 'bt8'

    def x(model, i):
        if i <= 3:
            return 1.0
        else:
            return 0.0

    model.x = aml.Var(aml.RangeSet(1, 5), initialize=x)

    def f_rule(model):
        return model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2
    model.f = aml.Objective(rule=f_rule)

    def con1(model):
        return model.x[1] - 1 - model.x[4] ** 2 + model.x[2] ** 2 == 0
    model.cons1 = aml.Constraint(rule=con1)

    def con2(model):
        return -1 + model.x[1] ** 2 + model.x[2] ** 2 - model.x[5] ** 2 == 0
    model.cons2 = aml.Constraint(rule=con2)
    return model


def create_model66():

    # bt9 LOR2-AN-4-2
    model = aml.ConcreteModel()
    model._name = 'bt9'

    model.S = aml.RangeSet(1, 4)
    model.x = aml.Var(model.S, initialize=2.0)

    def f(model):
        return -model.x[1]
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.x[2] - model.x[1] ** 3 - model.x[3] ** 2 == 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return -model.x[2] + model.x[1] ** 2 - model.x[4] ** 2 == 0
    model.cons2 = aml.Constraint(rule=cons2)
    return model


def create_model67():

    # byrdsphr LQR2-AN-3-2
    model = aml.ConcreteModel()
    model._name = 'byrdsphr'
    model.S = aml.RangeSet(1, 3)
    xinit = dict()
    xinit[1] = 5.0
    xinit[2] = 0.0001
    xinit[3] = -0.0001

    model.x = aml.Var(model.S, initialize=xinit)

    def f(model):
        return -model.x[1] - model.x[2] - model.x[3]
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -9.0 + model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 == 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return -9.0 + (model.x[1] - 1.0) ** 2 + model.x[2] ** 2 + model.x[3] ** 2 == 0
    model.cons2 = aml.Constraint(rule=cons2)

    return model


def create_model68():

    # camel6 OBR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'camel6'

    model.S = aml.RangeSet(1, 2)
    model.x = aml.Var(model.S, initialize=1.1)

    def f(model):
        return 4 * model.x[1] ** 2 - 2.1 * model.x[1] ** 4 + model.x[1] ** 6 / 3 + model.x[1] * model.x[2] - 4 * \
        model.x[2] ** 2 + 4 * model.x[2] ** 4

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return (-3, model.x[1], 3)

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return (-1.5, model.x[2], 1.5)

    model.cons2 = aml.Constraint(rule=cons2)
    return model


def create_model69():

    # cantilvr LOR2-MN-5-1
    model = aml.ConcreteModel()
    model._name = 'cantilvr'
    model.S = aml.RangeSet(1, 5)
    model.num = dict()
    model.num[1] = 61.0
    model.num[2] = 37.0
    model.num[3] = 19.0
    model.num[4] = 7.0
    model.num[5] = 1.0

    model.x = aml.Var(model.S, bounds=(0.000001, None), initialize=1.0)

    def f(model):
        se = sum([model.x[i] for i in model.S])
        return se * 0.0624
    model.f = aml.Objective(rule=f)

    def cons1(model):
        see = sum([model.num[i] / model.x[i] ** 3 for i in model.S])
        return see - 1.0 <= 0

    model.cons1 = aml.Constraint(rule=cons1)
    return model


def create_model70():

    # catena LQR2-AY-V-V
    model = aml.ConcreteModel()
    model._name = 'catena'
    model.N = aml.Param(initialize=10)

    model.gamma = aml.Param(initialize=9.81)
    model.tmass = aml.Param(initialize=500.0)
    model.bl = aml.Param(initialize=1.0)
    model.fract = aml.Param(initialize=0.6)

    def len_rule(model):
        return aml.value(model.bl) * (aml.value(model.N) + 1) * aml.value(model.fract)

    def mass_rule(model):
        return aml.value(model.tmass) / (aml.value(model.N) + 1.0)

    def mg_rule(model):
        return aml.value(model.mass) * aml.value(model.gamma)

    model.length = aml.Param(initialize=len_rule)
    model.mass = aml.Param(initialize=mass_rule)
    model.mg = aml.Param(initialize=mg_rule)

    model.Sv = aml.RangeSet(0, model.N + 1)
    model.So = aml.RangeSet(1, model.N)
    model.Sc = aml.RangeSet(1, model.N + 1)

    def x_rule(model, i):
        return i * aml.value(model.length) / (aml.value(model.N) + 1.0)

    def y_rule(model, i):
        return -i * aml.value(model.length) / (aml.value(model.N) + 1.0)

    model.x = aml.Var(model.Sv, initialize=x_rule)
    model.y = aml.Var(model.Sv, initialize=y_rule)
    model.z = aml.Var(model.Sv, initialize=0.0)

    def f(model):
        obsum = 0
        for i in model.So:
            obsum += aml.value(model.mg) * model.y[i]
        obsum += aml.value(model.mg) * model.y[aml.value(model.N) + 1] / 2.0
        expr = aml.value(model.mg) * model.y[0] / 2.0 + obsum
        return expr
    model.f = aml.Objective(rule=f)

    def cons1(model, i):
        expr = (model.x[i] - model.x[i - 1]) ** 2 + (model.y[i] - model.y[i - 1]) ** 2 + (model.z[i] - model.z[
            i - 1]) ** 2
        return expr == aml.value(model.bl) ** 2

    model.cons1 = aml.Constraint(model.Sc, rule=cons1)

    model.x[0] = 0.0
    model.x[0].fixed = True
    model.y[0] = 0.0
    model.y[0].fixed = True
    model.z[0] = 0.0
    model.z[0].fixed = True
    model.x[aml.value(model.N) + 1] = aml.value(model.length)
    model.x[aml.value(model.N) + 1].fixed = True

    return model


def create_model71():

    # catenary LQR2-AY-V-V
    model = aml.ConcreteModel()
    model._name = 'catenary'

    N = 165
    gamma = 9.81
    tmass = 500.0
    bl = 1.0
    fract = 0.6

    length = bl * (N + 1) * fract
    mass = tmass / (N + 1)
    mg = mass * gamma

    def x(model, i):
        return i * length / (N + 1)
    model.x = aml.Var(aml.RangeSet(0, N + 1), initialize=x)

    def y(model, i):
        return 0.0

    model.y = aml.Var(aml.RangeSet(0, N + 1), initialize=y)

    model.z = aml.Var(aml.RangeSet(0, N + 1), initialize=0.0)

    def f_rule(model):
        return mg * model.y[0] / 2.0 + sum(mg * model.y[i] for i in range(1, N + 1)) + mg * model.y[N + 1] / 2.0

    model.f = aml.Objective(rule=f_rule)

    def con1(model, i):
        return (model.x[i] - model.x[i - 1]) ** 2 + (model.y[i] - model.y[i - 1]) ** 2 + (model.x[i] - model.z[
            i - 1]) ** 2 == bl ** 2

    model.cons1 = aml.Constraint(aml.RangeSet(1, N + 1), rule=con1)

    model.x[0] = 0.0
    model.x[0].fixed = True
    model.y[0] = 0.0
    model.y[0].fixed = True
    model.z[0] = 0.0
    model.z[0].fixed = True
    model.x[N + 1] = length
    model.x[N + 1].fixed = True

    return model


# ToDo: try with line search
def create_model72():

    # cb2 LOR2-AN-3-3
    model = aml.ConcreteModel()
    model._name = 'cb2'

    model.S = aml.RangeSet(1, 2)
    model.x = aml.Var(model.S, initialize=2.0)
    model.u = aml.Var(initialize=1.0)

    def f(model):
        return model.u
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.u - model.x[1] ** 2 - model.x[2] ** 4 >= 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return model.u - (2.0 - model.x[1]) ** 2 - (2.0 - model.x[2]) ** 2 >= 0
    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return model.u - 2 * aml.exp(model.x[2] - model.x[1]) >= 0
    model.cons3 = aml.Constraint(rule=cons3)

    return model


# ToDo: try with line search
def create_model73():

    # cb3 LOR2-AN-3-3
    model = aml.ConcreteModel()
    model._name = 'cb3'

    model.S = aml.RangeSet(1, 2)
    model.x = aml.Var(model.S, initialize=2.0)
    model.u = aml.Var(initialize=1.0)

    def f(model):
        return model.u
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.u - model.x[1] ** 4 - model.x[2] ** 2 >= 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return model.u - (2.0 - model.x[1]) ** 2 - (2.0 - model.x[2]) ** 2 >= 0
    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return model.u - 2 * aml.exp(model.x[2] - model.x[1]) >= 0
    model.cons3 = aml.Constraint(rule=cons3)

    return model


def create_model74():

    # cbratu NOR2-MN-V-V
    model = aml.ConcreteModel()
    model._name = 'cbratu'

    p = 23
    l = 5.0
    h = 1.0 / (p - 1.0)
    c = h ** 2 / l
    model.u = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)
    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)

    model.f = aml.Objective(expr=0.0)
    def con1(model, i, j):
        return (4 * model.u[i, j] - model.u[i + 1, j] - model.u[i - 1, j] - model.u[i, j + 1] - \
                model.u[i, j - 1] - c * aml.exp(model.u[i, j]) * aml.cos(model.x[i, j])) == 0

    model.cons1 = aml.Constraint(aml.RangeSet(2, p - 1), aml.RangeSet(2, p - 1), rule=con1)

    def con2(model, i, j):
        return (4 * model.x[i, j] - model.x[i + 1, j] - model.x[i - 1, j] - model.x[i, j + 1] - \
                model.x[i, j - 1] - c * aml.exp(model.u[i, j]) * aml.sin(model.x[i, j])) == 0
    model.cons2 = aml.Constraint(aml.RangeSet(2, p - 1), aml.RangeSet(2, p - 1), rule=con2)

    for j in range(1, p + 1):
        model.u[1, j].fix(0.0)
        model.u[p, j].fix(0.0)
        model.x[1, j].fix(0.0)
        model.x[p, j].fix(0.0)

    for i in range(2, p):
        model.u[i, p].fix(0.0)
        model.u[i, 1].fix(0.0)
        model.x[i, p].fix(0.0)
        model.x[i, 1].fix(0.0)

    return model


def create_model75():

    # chaconn1 LOR2-AY-3-3
    model = aml.ConcreteModel()
    model._name = 'chaconn1'

    model.S = aml.RangeSet(1, 2)
    xinit = dict()
    xinit[1] = 1.0
    xinit[2] = -0.1
    model.x = aml.Var(model.S, initialize=xinit)
    model.u = aml.Var()

    def f(model):
        return model.u
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -model.u + model.x[1] ** 2 + model.x[2] ** 4 <= 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return -model.u + (2 - model.x[1]) ** 2 + (2 - model.x[2]) ** 2 <= 0
    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return -model.u + 2 * aml.exp(model.x[2] - model.x[1]) <= 0
    model.cons3 = aml.Constraint(rule=cons3)

    return model


def create_model76():

    # chacon2 LOR2-AY-3-3
    model = aml.ConcreteModel()
    model._name = 'chacon2'

    model.S = aml.RangeSet(1, 2)
    xinit = dict()
    xinit[1] = 1.0
    xinit[2] = -0.1
    model.x = aml.Var(model.S, initialize=xinit)
    model.u = aml.Var()

    def f(model):
        return model.u
    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -model.u + model.x[1] ** 4 + model.x[2] ** 2 <= 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return -model.u + (2 - model.x[1]) ** 2 + (2 - model.x[2]) ** 2 <= 0
    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return -model.u + 2 * aml.exp(model.x[2] - model.x[1]) <= 0
    model.cons3 = aml.Constraint(rule=cons3)

    return model


def create_model77():

    # chandheq NOR2-RN-V-V
    model = aml.ConcreteModel()
    model._name = 'chandheg'

    n = 100
    c = 1.0

    def x_init(model, i):
        return i / float(n)
    model.x = aml.Param(aml.RangeSet(1, n), initialize=x_init)

    def w_init(model, i):
        return 1.0 / float(n)

    model.w = aml.Param(aml.RangeSet(1, n), initialize=w_init)
    model.h = aml.Var(aml.RangeSet(1, n), initialize=1.0, bounds=(0, None))
    model.f = aml.Objective(expr=0.0)

    def con1(model, i):
        return sum(-0.5 * c * model.w[j] * model.x[i] / (model.x[i] + model.x[j]) * model.h[i] * model.h[j] for j in
                   range(1, n + 1)) + model.h[i] == 1.0

    model.cons = aml.Constraint(aml.RangeSet(1, n), rule=con1)

    return model


def create_model78():

    # chenhark QBR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'chenhark'

    n = 1000
    nfree = 500
    ndegen = 200

    def x_p_init(model, i):
        if i <= 0:
            return 0.0
        elif i > nfree:
            return 0.0
        else:
            return 1.0

    model.x_p = aml.Param(aml.RangeSet(-1, n + 2), initialize=x_p_init)
    def x_init(model, i):
        return 0.5
    model.x = aml.Var(aml.RangeSet(1, n), initialize=x_init, bounds=(0.0, None))

    def f(model):
        return sum(0.5 * (model.x[i + 1] + model.x[i - 1] - 2 * model.x[i]) ** 2 for i in range(2, n)) + \
               0.5 * model.x[1] ** 2 + 0.5 * (2 * model.x[1] - model.x[2]) ** 2 + 0.5 * (2 * model.x[n] - model.x[
            n - 1]) ** 2 + \
               0.5 * (model.x[n]) ** 2 + sum(model.x[i] * (-6 * model.x_p[i] + \
                                                           4 * model.x_p[i + 1] + 4 * model.x_p[i - 1] - model.x_p[
                                                               i + 2] - model.x_p[i - 2]) for i in
                                             range(1, nfree + ndegen + 1)) + \
               sum(model.x[i] * (-6 * model.x_p[i] + 4 * model.x_p[i + 1] + 4 * model.x_p[i - 1] - \
                                 model.x_p[i + 2] - model.x_p[i - 2] + 1) for i in range(nfree + ndegen + 1, n + 1))

    model.f = aml.Objective(rule=f)

    return model


# ToDo: converges but try with line search
def create_model79():

    # clnlbeam OOR2-MN-V-V
    model = aml.ConcreteModel()
    model._name = 'clnlbeam'

    ni = 500
    alpha = 350.0
    h = 1.0 / ni

    def t_init(model, i):
        return 0.05 * aml.cos(i * h)

    model.t = aml.Var(aml.RangeSet(0, ni), initialize=t_init, bounds=(-1.0, 1.0))

    def x_init(model, i):
        return 0.05 * aml.cos(i * h)

    model.x = aml.Var(aml.RangeSet(0, ni), initialize=x_init, bounds=(-0.05, 0.05))

    model.u = aml.Var(aml.RangeSet(0, ni))

    def f(model):
        return sum((0.5 * h * (model.u[i + 1] ** 2 + model.u[i] ** 2) + 0.5 * alpha * h * (aml.cos(model.t[i + 1]) + \
                                                                                           aml.cos(model.t[i]))) for i in
                   range(0, ni))

    model.f = aml.Objective(rule=f)

    def con1(model, i):
        return model.x[i + 1] - model.x[i] - 0.5 * h * (aml.sin(model.t[i + 1]) + aml.sin(model.t[i])) == 0

    model.cons1 = aml.Constraint(aml.RangeSet(0, ni - 1), rule=con1)

    def con2(model, i):
        return model.t[i + 1] - model.t[i] - 0.5 * h * model.u[i + 1] - 0.5 * h * model.u[i] == 0

    model.cons2 = aml.Constraint(aml.RangeSet(0, ni - 1), rule=con2)

    model.x[0] = 0.0
    model.x[0].fixed = True
    model.x[ni] = 0.0
    model.x[ni].fixed = True
    model.t[0] = 0.0
    model.t[0].fixed = True
    model.t[ni] = 0.0
    model.t[ni].fixed = True
    return model


# ToDo: works but with tolerance 1e-6
def create_model80():

    # clplateb OXR2-MN-V-0
    model = aml.ConcreteModel()
    model._name = 'clplateb'

    p = 71
    wght = -0.1
    disw = wght / (p - 1)
    hp2 = 0.5 * p ** 2

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + \
                   0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + \
                   hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4 + \
                   hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4 \
                   for i in range(2, p + 1) for j in range(2, p + 1)) + sum(
            wght * model.x[p, j] for j in range(1, p + 1))

    model.f = aml.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model


def create_model81():

    # clplatec OXR2-MN-V-0
    model = aml.ConcreteModel()
    model._name = 'clplatec'

    p = 71
    wght = -0.1
    hp2 = 0.5 * p ** 2
    wr = wght * 0.99
    wl = wght * 0.01

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + \
                   0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + \
                   (model.x[i, j] - model.x[i, j - 1]) ** 4 + \
                   (model.x[i, j] - model.x[i - 1, j]) ** 4 for i in range(2, p + 1) for j in range(2, p + 1)) + \
               (wr * model.x[p, p] + wl * model.x[p, 1])

    model.f = aml.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model


def create_model82():

    # cluster NOR2-AN-2-2
    model = aml.ConcreteModel()
    model._name = 'cluster'

    model.x1 = aml.Var()
    model.x2 = aml.Var()

    model.x1.value = 1.0
    model.x2.value = 1.0

    def obj_rule(model):
        return 0

    model.obj = aml.Objective(rule=obj_rule)

    def con1(model):
        return ((model.x1 - model.x2 * model.x2) * (model.x1 - aml.sin(model.x2))) == 0

    model.cons1 = aml.Constraint(rule=con1)

    def con2(model):
        return (((aml.cos(model.x2)) - model.x1) * (model.x2 - aml.cos(model.x1))) == 0

    model.cons2 = aml.Constraint(rule=con2)

    return model


def create_model83():

    # concon
    model = aml.ConcreteModel()
    model._name = 'concon'

    n = 7
    m = 4
    demand = -1000.0
    pmax1 = 914.73
    pmax2 = 904.73
    k = -0.597053452

    model.p1 = aml.Var(bounds=(None, 914.73), initialize=965)
    model.p2 = aml.Var(initialize=965)
    model.p3 = aml.Var(bounds=(None, 904.73), initialize=965)
    model.p4 = aml.Var(initialize=965)
    model.p5 = aml.Var(bounds=(None, 904.73), initialize=965)
    model.p6 = aml.Var(initialize=965)
    model.p7 = aml.Var(bounds=(None, 914.73), initialize=965)
    model.q1 = aml.Var(initialize=100.0)
    model.f1 = aml.Var(initialize=1000.0)
    model.q2 = aml.Var(initialize=100.0)
    model.f2 = aml.Var(initialize=1000.0)
    model.q3 = aml.Var(initialize=-100.0)
    model.f3 = aml.Var(initialize=1000.0)
    model.q4 = aml.Var(initialize=-100.0)
    model.f4 = aml.Var(bounds=(None, 400.0), initialize=1000.0)

    def obj_rule(model):
        return - model.p1 - model.p2 - model.p3 - model.p4 - model.p5 - model.p6 - model.p7

    model.obj = aml.Objective(rule=obj_rule)

    def pan1(model):
        return model.p1 * (abs(model.p1)) - model.p2 * (abs(model.p2)) - 0.597053452 * model.q1 * (abs(
            model.q1)) ** 0.8539 == 0

    def pan2(model):
        return model.p3 * (abs(model.p3)) - model.p4 * (abs(model.p4)) - 0.597053452 * model.q2 * (abs(
            model.q2)) ** 0.8539 == 0

    def pan3(model):
        return model.p4 * (abs(model.p4)) - model.p5 * (abs(model.p5)) - 0.597053452 * model.q3 * (abs(
            model.q3)) ** 0.8539 == 0

    def pan4(model):
        return model.p6 * (abs(model.p6)) - model.p7 * (abs(model.p7)) - 0.597053452 * model.q4 * (abs(
            model.q4)) ** 0.8539 == 0

    def m1(model):
        return model.q1 - model.f3 == 0

    def m2(model):
        return -model.q1 + model.f1 == 0

    def m3(model):
        return model.q2 - model.f1 == 0

    def m4(model):
        return -model.q2 + model.q3 + 1000.0 == 0

    def m5(model):
        return -model.q3 - model.f2 == 0

    def m6(model):
        return model.q4 + model.f2 == 0

    def m7(model):
        return -model.q4 - model.f4 == 0

    model.pan1 = aml.Constraint(rule=pan1)
    model.pan2 = aml.Constraint(rule=pan2)
    model.pan3 = aml.Constraint(rule=pan3)
    model.pan4 = aml.Constraint(rule=pan4)
    model.mbal1 = aml.Constraint(rule=m1)
    model.mbal2 = aml.Constraint(rule=m2)
    model.mbal3 = aml.Constraint(rule=m3)
    model.mbal4 = aml.Constraint(rule=m4)
    model.mbal5 = aml.Constraint(rule=m5)
    model.mbal6 = aml.Constraint(rule=m6)
    model.mbal7 = aml.Constraint(rule=m7)

    return model


# ToDo: try with line search
def create_model84():

    # congigmz LQR2-AN-3-5
    model = aml.ConcreteModel()
    model._name = 'congigmz'

    model.x = aml.Var(aml.RangeSet(1, 2), initialize=2.0)
    model.z = aml.Var(initialize=2.0)

    def f(model):
        return model.z

    model.f = aml.Objective(rule=f)

    def con1(model):
        return model.z + 5 * model.x[1] - model.x[2] >= 0

    model.cons1 = aml.Constraint(rule=con1)

    def con2(model):
        return model.z - 4 * model.x[2] - model.x[1] ** 2 - model.x[2] ** 2 >= 0

    model.cons2 = aml.Constraint(rule=con2)

    def con3(model):
        return model.z - 5 * model.x[1] - model.x[2] >= 0

    model.cons3 = aml.Constraint(rule=con3)

    def con4(model):
        return model.x[1] + model.x[2] + 10.0 <= 0

    model.cons4 = aml.Constraint(rule=con4)

    def con5(model):
        return 2 * model.x[1] ** 2 - model.x[2] ** 2 + 4.0 <= 0

    model.cons5 = aml.Constraint(rule=con5)

    return model


# ToDo: try with line search
def create_model85():

    # corckscrw SOR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'corckscrw'

    model.t = aml.Param(initialize=1000)
    model.xt = aml.Param(initialize=10.0)
    model.mass = aml.Param(initialize=0.37)
    model.tol = aml.Param(initialize=0.1)

    def h_rule(model):
        return aml.value(model.xt) / aml.value(model.t)

    model.h = aml.Param(initialize=h_rule)

    def w_rule(model):
        return aml.value(model.xt) * (aml.value(model.t) + 1.0) / 2.0

    model.w = aml.Param(initialize=w_rule)

    def fmax_rule(model):
        return aml.value(model.xt) / aml.value(model.t)

    model.fmax = aml.Param(initialize=fmax_rule)

    model.S = aml.RangeSet(0, model.t)
    model.SS = aml.RangeSet(1, model.t)

    def x_rule(model, i):
        return i * aml.value(model.h)

    def x_bound(model, i):
        l = 0.0
        u = aml.value(model.xt)
        return (l, u)

    model.x = aml.Var(model.S, bounds=x_bound, initialize=x_rule)
    model.y = aml.Var(model.S)
    model.z = aml.Var(model.S)
    model.vx = aml.Var(model.S, initialize=1.0)
    model.vy = aml.Var(model.S)
    model.vz = aml.Var(model.S)

    def u_bound(model, i):
        l = -aml.value(model.fmax)
        u = aml.value(model.fmax)
        return (l, u)

    model.ux = aml.Var(model.SS, bounds=u_bound)
    model.uy = aml.Var(model.SS, bounds=u_bound)
    model.uz = aml.Var(model.SS, bounds=u_bound)

    model.x[0] = 0.0
    model.x[0].fixed = True
    model.y[0] = 0.0
    model.y[0].fixed = True
    model.z[0] = 1.0
    model.z[0].fixed = True
    model.vx[0] = 0.0
    model.vx[0].fixed = True
    model.vy[0] = 0.0
    model.vy[0].fixed = True
    model.vz[0] = 0.0
    model.vz[0].fixed = True
    model.vx[aml.value(model.t)] = 0.0
    model.vx[aml.value(model.t)].fixed = True
    model.vy[aml.value(model.t)] = 0.0
    model.vy[aml.value(model.t)].fixed = True
    model.vz[aml.value(model.t)] = 0.0
    model.vz[aml.value(model.t)].fixed = True

    def f(model):
        return sum([(i * aml.value(model.h) / aml.value(model.w)) * (model.x[i] - aml.value(model.xt)) ** 2 for i in model.SS])

    model.f = aml.Objective(rule=f)

    def acx(model, i):
        return aml.value(model.mass) * (model.vx[i] - model.vx[i - 1]) / aml.value(model.h) - model.ux[i] == 0

    model.acx = aml.Constraint(model.SS, rule=acx)

    def acy(model, i):
        return aml.value(model.mass) * (model.vy[i] - model.vy[i - 1]) / aml.value(model.h) - model.uy[i] == 0

    model.acy = aml.Constraint(model.SS, rule=acy)

    def acz(model, i):
        return aml.value(model.mass) * (model.vz[i] - model.vz[i - 1]) / aml.value(model.h) - model.uz[i] == 0

    model.acz = aml.Constraint(model.SS, rule=acz)

    def psx(model, i):
        return (model.x[i] - model.x[i - 1]) / aml.value(model.h) - model.vx[i] == 0

    model.psx = aml.Constraint(model.SS, rule=psx)

    def psy(model, i):
        return (model.y[i] - model.y[i - 1]) / aml.value(model.h) - model.vy[i] == 0

    model.psy = aml.Constraint(model.SS, rule=psy)

    def psz(model, i):
        return (model.z[i] - model.z[i - 1]) / aml.value(model.h) - model.vz[i] == 0

    model.psz = aml.Constraint(model.SS, rule=psz)

    def sc(model, i):
        return (model.y[i] - aml.sin(model.x[i])) ** 2 + (model.z[i] - aml.cos(model.x[i])) ** 2 - aml.value(model.tol) ** 2 <= 0

    model.sc = aml.Constraint(model.SS, rule=sc)
    return model


def create_model86():

    # cosine OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'cosine'

    N = 10000

    model.x = aml.Var(aml.RangeSet(1, N), initialize=1.0)

    def f_rule(model):
        return sum(aml.cos(-0.5 * model.x[i + 1] + model.x[i] ** 2) for i in range(1, N))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model87():

    # cragglvy OUR2-AY-V-0
    model = aml.ConcreteModel()
    model._name = 'cragglvy'

    m = 2499
    n = 2 * m + 2

    def x(model, i):
        if i == 1:
            return 1.0
        else:
            return 2.0

    model.x = aml.Var(aml.RangeSet(1, n), initialize=x)

    def f(model):
        return sum((aml.exp(model.x[2 * i - 1]) - model.x[2 * i]) ** 4 + \
                   100 * (model.x[2 * i] - model.x[2 * i + 1]) ** 6 + \
                   (aml.tan(model.x[2 * i + 1] - model.x[2 * i + 2]) + model.x[2 * i + 1] - model.x[2 * i + 2]) ** 4 + \
                   (model.x[2 * i - 1]) ** 8 + \
                   (model.x[2 * i + 2] - 1.0) ** 2 for i in range(1, m + 1))

    model.f = aml.Objective(rule=f)
    return model


# ToDo: try with line search
def create_model88():

    #cresc4 OOR2-MY-6-8
    model = aml.ConcreteModel()
    model._name = 'cresc4'

    np = 4
    model.x = dict()
    model.x[1] = 1.0
    model.x[2] = 0.0
    model.x[3] = 0.0
    model.x[4] = 0.5

    model.y = dict()
    model.y[1] = 0.0
    model.y[2] = 1.0
    model.y[3] = -1.0
    model.y[4] = 0.0

    model.v1 = aml.Var(initialize=-40.0)
    model.w1 = aml.Var(initialize=5.0)
    model.d = aml.Var(bounds=(1e-8, None), initialize=1.0)
    model.a = aml.Var(bounds=(1.0, None), initialize=2.0)
    model.t = aml.Var(bounds=(0.0, 6.2831852), initialize=1.5)
    model.r = aml.Var(bounds=(0.39, None), initialize=0.75)

    def f(model):
        return (model.d + model.r) ** 2 * aml.acos(-((model.a * model.d) ** 2 - (model.a * model.d + model.r) ** 2 \
                                                 + (model.d + model.r) ** 2) / (
                                               2 * (model.d + model.r) * model.a * model.d)) \
               - (model.a * model.d + model.r) ** 2 * aml.acos(
            ((model.a * model.d) ** 2 + (model.a * model.d + model.r) ** 2 - \
             (model.d + model.r) ** 2) / (2 * (model.a * model.d + model.r) * model.a * model.d)) \
               + (model.d + model.r) * model.a * model.d * aml.sin(
            aml.acos(-((model.a * model.d) ** 2 - (model.a * model.d + model.r) ** 2 \
                   + (model.d + model.r) ** 2) / (2 * (model.d + model.r) * model.a * model.d)))

    model.f = aml.Objective(rule=f)

    def con1(model, i):
        return (model.v1 + model.a * model.d * aml.cos(model.t) - model.x[i]) ** 2 + \
               (model.w1 + model.a * model.d * aml.sin(model.t) - model.y[i]) ** 2 - (model.d + model.r) ** 2 <= 0.0
    model.cons1 = aml.Constraint(aml.RangeSet(1, np), rule=con1)

    def con2(model, i):
        return (model.v1 - model.x[i]) ** 2 + (model.w1 - model.y[i]) ** 2 - (model.a * model.d + model.r) ** 2 >= 0.0
    model.cons2 = aml.Constraint(aml.RangeSet(1, np), rule=con2)

    return model


def create_model89():

    # csfi1 LOR2-RN-5-4
    model = aml.ConcreteModel()
    model._name = 'csfi1'

    density = 0.284
    lenmax = 60.0
    maxaspr = 2.0
    minthick = 7.0
    minarea = 200.0
    maxarea = 250.0
    k = 1.0

    model.thick = aml.Var(bounds=(minthick, None), initialize=0.5)
    model.wid = aml.Var(bounds=(0.0, None), initialize=0.5)
    model.len = aml.Var(bounds=(0.0, lenmax), initialize=0.5)
    model.tph = aml.Var(bounds=(0.0, None), initialize=0.5)
    model.ipm = aml.Var(bounds=(0.0, None), initialize=0.5)

    def f(model):
        return -model.tph

    model.f = aml.Objective(rule=f)

    def con1(model):
        return 117.370892 * model.tph / (model.wid * model.thick) - model.ipm == 0.0

    def con2(model):
        return model.thick ** 2 * model.ipm / 48.0 - model.len == 0.0;

    def con3(model):
        return model.wid / model.thick <= maxaspr

    def con4(model):
        return 0.0 <= model.thick * model.wid - minarea <= maxarea - minarea

    model.cons1 = aml.Constraint(rule=con1)
    model.cons2 = aml.Constraint(rule=con2)
    model.cons3 = aml.Constraint(rule=con3)
    model.cons4 = aml.Constraint(rule=con4)
    return model


# ToDo: try with line search
def create_model90():

    # csfi2 LOR2-RN-5-4
    model = aml.ConcreteModel()
    model._name = 'csfi2'

    model.mintph = aml.Param(initialize=45.0)
    model.minthick = aml.Param(initialize=7.0)
    model.minarea = aml.Param(initialize=200.0)
    model.maxarea = aml.Param(initialize=250.0)
    model.maxaspr = aml.Param(initialize=2.0)
    model.k = aml.Param(initialize=1.0)

    def thick_bound(model):
        l = aml.value(model.minthick)
        return (l, None)

    model.thick = aml.Var(bounds=thick_bound, initialize=0.5)
    model.wid = aml.Var(bounds=(0.0, None), initialize=0.5)
    model.len = aml.Var(bounds=(0.0, None), initialize=0.5)

    def tph_bound(model):
        l = aml.value(model.mintph)
        return (l, None)

    model.tph = aml.Var(bounds=tph_bound, initialize=0.5)
    model.ipm = aml.Var(bounds=(0.0, None), initialize=0.5)

    def f(model):
        return model.len

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return 117.370892 * model.tph / (model.wid * model.thick) - model.ipm == 0.0

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return model.thick ** 2 * model.ipm / 48.0 - model.len == 0.0

    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return model.wid / model.thick <= aml.value(model.maxaspr)

    model.cons3 = aml.Constraint(rule=cons3)

    def cons4(model):
        # return 0.0 <= model.thick*model.wid - value(model.minarea) <= value(model.maxarea) - value(model.minarea)
        return (0.0, model.thick * model.wid - aml.value(model.minarea), aml.value(model.maxarea) - aml.value(model.minarea))

    model.cons4 = aml.Constraint(rule=cons4)

    return model


def create_model91():

    # cube SUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'cube'

    N = 2
    xinit = dict()
    xinit[1] = -1.2
    xinit[2] = 1.0
    model.x = aml.Var(aml.RangeSet(1, 2), initialize=xinit)

    def f(model):
        return (model.x[1] - 1.0) ** 2 + sum(100 * (model.x[i] - model.x[i - 1] ** 3) ** 2 for i in range(2, N + 1))

    model.f = aml.Objective(rule=f)
    return model


def create_model92():

    # curly10 SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'curly10'

    N = 10000
    K = 10

    model.x = aml.Var(aml.RangeSet(1, N), initialize=0.0001 / (N + 1))

    def Q(model, i):
        if i <= N - K:
            return sum(model.x[j] for j in range(i, i + K + 1))
        else:
            return sum(model.x[j] for j in range(i, N + 1))

    def f(model):
        return sum(Q(model, i) * (Q(model, i) * (Q(model, i) ** 2 - 20) - 0.1) for i in range(1, N + 1))

    model.f = aml.Objective(rule=f)

    return model


def create_model93():

    # curly20 SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'curly20'

    N = 10000
    K = 20

    model.x = aml.Var(aml.RangeSet(1, N), initialize=0.0001 / (N + 1))

    def Q(model, i):
        if i <= N - K:
            return sum(model.x[j] for j in range(i, i + K + 1))
        else:
            return sum(model.x[j] for j in range(i, N + 1))

    def f(model):
        return sum(Q(model, i) * (Q(model, i) * (Q(model, i) ** 2 - 20) - 0.1) for i in range(1, N + 1))

    model.f = aml.Objective(rule=f)

    return model


def create_model94():

    # curly30 SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'curly30'

    N = 10000
    K = 30

    model.x = aml.Var(aml.RangeSet(1, N), initialize=0.0001 / (N + 1))

    def Q(model, i):
        if i <= N - K:
            return sum(model.x[j] for j in range(i, i + K + 1))
        else:
            return sum(model.x[j] for j in range(i, N + 1))

    def f(model):
        return sum(Q(model, i) * (Q(model, i) * (Q(model, i) ** 2 - 20) - 0.1) for i in range(1, N + 1))

    model.f = aml.Objective(rule=f)

    return model


def create_model95():

    # cvxqp3 QLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'cvxqp3'
    N = 10000
    model.N = aml.Param(initialize=N)
    model.M = aml.Param(initialize=3 * N / 4)

    model.S = aml.RangeSet(1, model.N)
    model.SS = aml.RangeSet(1, model.M)
    #model.x = aml.Var(model.S, initialize=0.5, bounds=(0.1, 10.0))
    model.x = aml.Var(model.S, initialize=0.5)

    def f(model):
        return sum([(model.x[i] + model.x[((2 * i - 1) % aml.value(model.N)) + 1] + model.x[
            ((3 * i - 1) % aml.value(model.N)) + 1]) ** 2 * i / 2.0 for i in model.S])

    model.f = aml.Objective(rule=f)

    def cons1(model, i):
        return model.x[i] + 2 * model.x[((4 * i - 1) % aml.value(model.N)) + 1] + 3 * model.x[
            ((5 * i - 1) % aml.value(model.N)) + 1] - 6.0 == 0

    model.cons1 = aml.Constraint(model.SS, rule=cons1)

    return model


def create_model96():

    # degenlpa LLR2-AN-20-15
    model = aml.ConcreteModel()
    model._name = 'degenlpa'

    model.N = aml.Param(initialize=20)
    model.M = aml.Param(initialize=15)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, bounds=(0.0, 1.0), initialize=1.0)

    def f(model):
        return 0.01 * model.x[2] + 33.333 * model.x[3] + 100.0 * model.x[4] + 0.01 * model.x[5] + 33.343 * model.x[
            6] + 100.01 * model.x[7] + 33.333 * model.x[8] + 133.33 * model.x[9] + 100.0 * model.x[10]

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -0.70785 + model.x[1] + 2 * model.x[2] + 2 * model.x[3] + 2 * model.x[4] + model.x[5] + 2 * model.x[
            6] + 2 * model.x[7] + model.x[8] + 2 * model.x[9] + model.x[10] == 0

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return 0.326 * model.x[1] - 101 * model.x[2] + 200 * model.x[5] + 0.06 * model.x[6] + 0.02 * model.x[7] == 0

    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return 0.0066667 * model.x[1] - 1.03 * model.x[3] + 200 * model.x[6] + 0.06 * model.x[8] + 0.02 * model.x[
            9] == 0

    model.cons3 = aml.Constraint(rule=cons3)

    def cons4(model):
        return 0.00066667 * model.x[1] - 1.01 * model.x[4] + 200 * model.x[7] + 0.06 * model.x[9] + 0.02 * model.x[
            10] == 0

    model.cons4 = aml.Constraint(rule=cons4)

    def cons5(model):
        return 0.978 * model.x[2] - 201 * model.x[5] + 100 * model.x[11] + 0.03 * model.x[12] + 0.01 * model.x[13] == 0

    model.cons5 = aml.Constraint(rule=cons5)

    def cons6(model):
        return 0.01 * model.x[2] + 0.489 * model.x[3] - 101.03 * model.x[6] + 100 * model.x[12] + 0.03 * model.x[
            14] + 0.01 * model.x[15] == 0

    model.cons6 = aml.Constraint(rule=cons6)

    def cons7(model):
        return 0.001 * model.x[2] + 0.489 * model.x[4] - 101.03 * model.x[7] + 100 * model.x[13] + 0.03 * model.x[
            15] + 0.01 * model.x[16] == 0

    model.cons7 = aml.Constraint(rule=cons7)

    def cons8(model):
        return 0.001 * model.x[3] + 0.01 * model.x[4] - 1.04 * model.x[9] + 100 * model.x[15] + 0.03 * model.x[
            18] + 0.01 * model.x[19] == 0

    model.cons8 = aml.Constraint(rule=cons8)

    def cons9(model):
        return 0.02 * model.x[3] - 1.06 * model.x[8] + 100 * model.x[14] + 0.03 * model.x[17] + 0.01 * model.x[19] == 0

    model.cons9 = aml.Constraint(rule=cons9)

    def cons10(model):
        return 0.002 * model.x[4] - 1.02 * model.x[10] + 100 * model.x[16] + 0.03 * model.x[19] + 0.01 * model.x[
            20] == 0

    model.cons10 = aml.Constraint(rule=cons10)

    def cons11(model):
        return -2.5742e-6 * model.x[11] + 0.00252 * model.x[13] - 0.61975 * model.x[16] + 1.03 * model.x[20] == 0

    model.cons11 = aml.Constraint(rule=cons11)

    def cons12(model):
        return -0.00257 * model.x[11] + 0.25221 * model.x[12] - 6.2 * model.x[14] + 1.09 * model.x[17] == 0

    model.cons12 = aml.Constraint(rule=cons12)

    def cons13(model):
        return 0.00629 * model.x[11] - 0.20555 * model.x[12] - 4.1106 * model.x[13] + 101.04 * model.x[15] + 505.1 * \
        model.x[16] - 256.72 * model.x[19] == 0

    model.cons13 = aml.Constraint(rule=cons13)

    def cons14(model):
        return 0.00841 * model.x[12] - 0.08406 * model.x[13] - 0.20667 * model.x[14] + 20.658 * model.x[16] + 1.07 * \
        model.x[18] - 10.5 * model.x[19] == 0

    model.cons14 = aml.Constraint(rule=cons14)

    return model


def create_model97():

    # degenlpb LLR2-AN-20-15
    model = aml.ConcreteModel()
    model._name = 'degenlpb'

    model.N = aml.Param(initialize=20)
    model.M = aml.Param(initialize=15)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, bounds=(0.0, 1.0), initialize=1.0)

    def f(model):
        return -1 * (
        0.01 * model.x[2] + 33.333 * model.x[3] + 100.0 * model.x[4] + 0.01 * model.x[5] + 33.343 * model.x[
            6] + 100.01 * model.x[7] + 33.333 * model.x[8] + 133.33 * model.x[9] + 100.0 * model.x[10])

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -0.70785 + model.x[1] + 2 * model.x[2] + 2 * model.x[3] + 2 * model.x[4] + model.x[5] + 2 * model.x[
            6] + 2 * model.x[7] + model.x[8] + 2 * model.x[9] + model.x[10] == 0

    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return 0.326 * model.x[1] - 101 * model.x[2] + 200 * model.x[5] + 0.06 * model.x[6] + 0.02 * model.x[7] == 0

    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return 0.0066667 * model.x[1] - 1.03 * model.x[3] + 200 * model.x[6] + 0.06 * model.x[8] + 0.02 * model.x[9] == 0

    model.cons3 = aml.Constraint(rule=cons3)

    def cons4(model):
        return 0.00066667 * model.x[1] - 1.01 * model.x[4] + 200 * model.x[7] + 0.06 * model.x[9] + 0.02 * model.x[10] == 0

    model.cons4 = aml.Constraint(rule=cons4)

    def cons5(model):
        return 0.978 * model.x[2] - 201 * model.x[5] + 100 * model.x[11] + 0.03 * model.x[12] + 0.01 * model.x[13] == 0

    model.cons5 = aml.Constraint(rule=cons5)

    def cons6(model):
        return 0.01 * model.x[2] + 0.489 * model.x[3] - 101.03 * model.x[6] + 100 * model.x[12] + \
        0.03 * model.x[14] + 0.01 * model.x[15] == 0

    model.cons6 = aml.Constraint(rule=cons6)

    def cons7(model):
        return 0.001 * model.x[2] + 0.489 * model.x[4] - 101.03 * model.x[7] + 100 * model.x[13] + 0.03 * model.x[15] + 0.01 * model.x[16] == 0

    model.cons7 = aml.Constraint(rule=cons7)

    def cons8(model):
        return 0.001 * model.x[3] + 0.01 * model.x[4] - 1.04 * model.x[9] + 100 * model.x[15] + \
        0.03 * model.x[18] + 0.01 * model.x[19] == 0

    model.cons8 = aml.Constraint(rule=cons8)

    def cons9(model):
        return 0.02 * model.x[3] - 1.06 * model.x[8] + 100 * model.x[14] + 0.03 * model.x[17] + 0.01 * model.x[19] == 0

    model.cons9 = aml.Constraint(rule=cons9)

    def cons10(model):
        return 0.002 * model.x[4] - 1.02 * model.x[10] + 100 * model.x[16] + 0.03 * model.x[19] + 0.01 * model.x[
            20] == 0

    model.cons10 = aml.Constraint(rule=cons10)

    def cons11(model):
        return -2.5742e-6 * model.x[11] + 0.00252 * model.x[13] - 0.61975 * model.x[16] + 1.03 * model.x[20] == 0

    model.cons11 = aml.Constraint(rule=cons11)

    def cons12(model):
        return -0.00257 * model.x[11] + 0.25221 * model.x[12] - 6.2 * model.x[14] + 1.09 * model.x[17] == 0

    model.cons12 = aml.Constraint(rule=cons12)

    def cons13(model):
        return 0.00629 * model.x[11] - 0.20555 * model.x[12] - 4.1106 * model.x[13] + 101.04 * model.x[15] + 505.1 * \
        model.x[16] - 256.72 * model.x[19] == 0

    model.cons13 = aml.Constraint(rule=cons13)

    def cons14(model):
        return 0.00841 * model.x[12] - 0.08406 * model.x[13] - 0.20667 * model.x[14] + 20.658 * model.x[16] + 1.07 * \
        model.x[18] - 10.5 * model.x[19] == 0

    model.cons14 = aml.Constraint(rule=cons14)

    def cons15(model):
        return -model.x[1] + 300 * model.x[2] + 0.09 * model.x[3] + 0.03 * model.x[4] == 0

    model.cons15 = aml.Constraint(rule=cons15)

    return model


# ToDo: try with line search
def create_model98():

    # demymalo LQR2-AN-3-3
    model = aml.ConcreteModel()
    model._name = 'demymalo'

    model.S = aml.RangeSet(1, 2)
    xinit = 1.0

    model.x = aml.Var(model.S, initialize=xinit)
    model.u = aml.Var()

    def f(model):
        return model.u

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return -model.u + 5 * model.x[1] + model.x[2] <= 0
    model.cons1 = aml.Constraint(rule=cons1)

    def cons2(model):
        return -model.u - 5 * model.x[1] + model.x[2] <= 0
    model.cons2 = aml.Constraint(rule=cons2)

    def cons3(model):
        return -model.u + 4 * model.x[2] + model.x[1] ** 2 + model.x[2] ** 2 <= 0
    model.cons3 = aml.Constraint(rule=cons3)

    return model


def create_model99():

    # denschna OUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'denschna'

    model.S = aml.RangeSet(1, 2)
    model.x = aml.Var(model.S, initialize=1.0)

    def f(model):
        return model.x[1] ** 4 + (model.x[1] + model.x[2]) ** 2 + (-1.0 + aml.exp(model.x[2])) ** 2

    model.f = aml.Objective(rule=f)
    return model


def create_model100():

    # denschnb SUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'denschnb'

    model.S = aml.RangeSet(1, 2)
    model.x = aml.Var(model.S, initialize=1.0)

    def f(model):
        return (model.x[1] - 2.0) ** 2 + ((model.x[1] - 2.0) * model.x[2]) ** 2 + (model.x[2] + 1.0) ** 2

    model.f = aml.Objective(rule=f)

    return model


def create_model101():

    # denschnc SUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'denschnc'

    model.S = aml.RangeSet(1, 2)
    xinit = dict()
    xinit[1] = 2.0
    xinit[2] = 3.0
    model.x = aml.Var(model.S, initialize=xinit)

    def f(model):
        return (-2 + model.x[1] ** 2 + model.x[2] ** 2) ** 2 + (-2 + aml.exp(model.x[1] - 1) + model.x[2] ** 3) ** 2

    model.f = aml.Objective(rule=f)
    return model


def create_model102():

    # dixmaanna OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dixmaanna'
    M = 1000
    model.M = aml.Param(initialize=M)
    model.N = aml.Param(initialize=3 * M)

    model.alpha = aml.Param(initialize=1.0)
    model.beta = aml.Param(initialize=0.0)
    model.gamma = aml.Param(initialize=0.125)
    model.delta = aml.Param(initialize=0.125)

    model.S1 = aml.RangeSet(1, 4)
    model.S2 = aml.RangeSet(1, model.N)
    model.K = aml.Param(model.S1, initialize=0)
    model.x = aml.Var(model.S2, initialize=2.0)

    model.S3 = aml.RangeSet(1, model.N - 1)
    model.S4 = aml.RangeSet(1, 2 * model.M)
    model.S5 = aml.RangeSet(1, model.M)

    def f(model):
        exp1 = sum([aml.value(model.alpha) * model.x[i] ** 2 * (i / aml.value(model.N)) ** aml.value(model.K[1]) for i in model.S2])
        exp2 = sum([aml.value(model.beta) * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (
        i / aml.value(model.N)) ** aml.value(model.K[2]) for i in model.S3])
        exp3 = sum([aml.value(model.gamma) * model.x[i] ** 2 * model.x[i + aml.value(model.M)] ** 4 * (
        i / aml.value(model.N)) ** aml.value(model.K[3]) for i in model.S4])
        exp4 = sum([aml.value(model.delta) * model.x[i] * model.x[i + 2 * aml.value(model.M)] * (i / aml.value(model.N)) ** aml.value(
            model.K[4]) for i in model.S5])
        return 1.0 + exp1 + exp2 + exp3 + exp4

    model.f = aml.Objective(rule=f)

    return model


def create_model103():

    # dixmaanb OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dixmaanb'
    M = 1000
    model.M = aml.Param(initialize=M)
    model.N = aml.Param(initialize=3 * M)

    model.alpha = aml.Param(initialize=1.0)
    model.beta = aml.Param(initialize=0.0625)
    model.gamma = aml.Param(initialize=0.0625)
    model.delta = aml.Param(initialize=0.0625)

    model.S1 = aml.RangeSet(1, 4)
    model.S2 = aml.RangeSet(1, model.N)
    model.K = aml.Param(model.S1, initialize=0)
    model.x = aml.Var(model.S2, initialize=2.0)

    model.S3 = aml.RangeSet(1, model.N - 1)
    model.S4 = aml.RangeSet(1, 2 * model.M)
    model.S5 = aml.RangeSet(1, model.M)

    def f(model):
        exp1 = sum([aml.value(model.alpha) * model.x[i] ** 2 * (i / aml.value(model.N)) ** aml.value(model.K[1]) for i in model.S2])
        exp2 = sum([aml.value(model.beta) * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (
        i / aml.value(model.N)) ** aml.value(model.K[2]) for i in model.S3])
        exp3 = sum([aml.value(model.gamma) * model.x[i] ** 2 * model.x[i + aml.value(model.M)] ** 4 * (
        i / aml.value(model.N)) ** aml.value(model.K[3]) for i in model.S4])
        exp4 = sum([aml.value(model.delta) * model.x[i] * model.x[i + 2 * aml.value(model.M)] * (i / aml.value(model.N)) ** aml.value(
            model.K[4]) for i in model.S5])
        return 1.0 + exp1 + exp2 + exp3 + exp4

    model.f = aml.Objective(rule=f)

    return model


def create_model104():

    # dixmaanc LQI2-RN-157-134
    model = aml.ConcreteModel()
    model._name = 'dixmaanc'

    M = 1000
    N = 3 * M

    alpha = 1.0
    beta = 0.125
    gamma = 0.125
    delta = 0.125

    model.K = aml.Param(aml.RangeSet(1, 4), initialize=0)
    model.x = aml.Var(aml.RangeSet(1, N), initialize=2.0)

    def f_rule(model):
        return 1.0 + sum(alpha * model.x[i] ** 2 * (i / N) ** model.K[1] for i in range(1, N + 1)) + \
               sum(beta * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (i / N) ** model.K[2] for i in
                   range(1, N)) + \
               sum(gamma * model.x[i] ** 2 * model.x[i + M] ** 4 * (i / N) ** model.K[3] for i in range(1, 2 * M + 1)) + \
               sum(delta * model.x[i] * model.x[i + 2 * M] * (i / N) ** model.K[4] for i in range(1, M + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model105():

    #dixmaand LQI2-RN-157-134
    model = aml.ConcreteModel()
    model._name = 'dixmaand'

    M = 1000
    N = 3 * M

    alpha = 1.0
    beta = 0.26
    gamma = 0.26
    delta = 0.26

    model.K = aml.Param(aml.RangeSet(1, 4), initialize=0)
    model.x = aml.Var(aml.RangeSet(1, N), initialize=2.0)

    def f_rule(model):
        return 1.0 + sum(alpha * model.x[i] ** 2 * (i / N) ** model.K[1] for i in range(1, N + 1)) + \
               sum(beta * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (i / N) ** model.K[2] for i in
                   range(1, N)) + \
               sum(gamma * model.x[i] ** 2 * model.x[i + M] ** 4 * (i / N) ** model.K[3] for i in range(1, 2 * M + 1)) + \
               sum(delta * model.x[i] * model.x[i + 2 * M] * (i / N) ** model.K[4] for i in range(1, M + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


def create_model106():

    # dixmaane OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dixmaane'

    M = 1000
    N = 3 * M

    alpha = 1.0
    beta = 0.0
    gamma = 0.125
    delta = 0.125

    model.K = dict()
    model.K[1] = 1.0
    model.K[2] = 0.0
    model.K[3] = 0.0
    model.K[4] = 1.0
    model.x = aml.Var(aml.RangeSet(1, N), initialize=2.0)

    def f_rule(model):
        return 1.0 + sum(alpha * model.x[i] ** 2 * (i / float(N)) ** model.K[1] for i in range(1, N + 1)) + \
               sum(beta * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (i / float(N)) ** model.K[2]
                   for i in range(1, N)) + \
               sum(gamma * model.x[i] ** 2 * model.x[i + M] ** 4 * (i / float(N)) ** model.K[3] for i in
                   range(1, 2 * M + 1)) + \
               sum(delta * model.x[i] * model.x[i + 2 * M] * (i / float(N)) ** model.K[4] for i in range(1, M + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model107():

    # dixmaanf OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dixmaanf'

    M = 1000
    N = 3 * M

    alpha = 1.0
    beta = 0.0625
    gamma = 0.0625
    delta = 0.0625

    model.K = dict()
    model.K[1] = 1.0
    model.K[2] = 0.0
    model.K[3] = 0.0
    model.K[4] = 1.0
    model.x = aml.Var(aml.RangeSet(1, N), initialize=2.0)

    def f_rule(model):
        return 1.0 + sum(alpha * model.x[i] ** 2 * (i / float(N)) ** model.K[1] for i in range(1, N + 1)) + \
               sum(beta * model.x[i] ** 2 * (model.x[i + 1] + model.x[i + 1] ** 2) ** 2 * (i / float(N)) ** model.K[2]
                   for i in range(1, N)) + \
               sum(gamma * model.x[i] ** 2 * model.x[i + M] ** 4 * (i / float(N)) ** model.K[3] for i in
                   range(1, 2 * M + 1)) + \
               sum(delta * model.x[i] * model.x[i + 2 * M] * (i / float(N)) ** model.K[4] for i in range(1, M + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model108():

    # dixon3dq QUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dixon3dq'

    model.n = aml.Param(initialize=10)
    model.S = aml.RangeSet(1, model.n)

    model.x = aml.Var(model.S, initialize=-1.0)

    model.SS = aml.RangeSet(2, model.n - 1)

    # rvdb comment: the sum should start at 1.
    def obj(model):
        es = sum([(model.x[j] - model.x[j + 1]) ** 2 for j in model.SS])
        return (model.x[1] - 1.0) ** 2 + es + (model.x[aml.value(model.n)] - 1.0) ** 2

    model.obj = aml.Objective(rule=obj)

    return model


# ToDo: fails for evaluation error
def create_model109():

    # djtl OUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'djtl'

    model.x1 = aml.Var(initialize=15.0)
    model.x2 = aml.Var(initialize=-1.0)

    value = aml.value
    def f_rule(model):
        if (-(value(model.x1) - 5) ** 2 - (value(model.x2) - 5) ** 2 + 200 + 1 <= 0.0):
            obj1 = (1E10 * (-(model.x1 - 5) ** 2 - (model.x2 - 5) ** 2 + 200) ** 2)
        else:
            obj1 = (-aml.log(-(model.x1 - 5) ** 2 - (model.x2 - 5) ** 2 + 200 + 1))
        if ((value(model.x1) - 5) ** 2 + (value(model.x2) - 5) ** 2 - 100 + 1 <= 0.0):
            obj2 = (1E10 * ((model.x1 - 5) ** 2 + (model.x2 - 5) ** 2 - 100) ** 2)
        else:
            obj2 = (-aml.log((model.x1 - 5) ** 2 + (model.x2 - 5) ** 2 - 100 + 1))
        if ((value(model.x2) - 5) ** 2 + (value(model.x1) - 6) ** 2 + 1 <= 0.0):
            obj3 = (1E10 * ((model.x2 - 5) ** 2 + (model.x1 - 6) ** 2) ** 2)
        else:
            obj3 = (-aml.log((model.x2 - 5) ** 2 + (model.x1 - 6) ** 2 + 1))
        if (-(value(model.x2) - 5) ** 2 - (value(model.x1) - 6) ** 2 + 82.81 + 1 <= 0.0):
            obj4 = (1E10 * (-(model.x2 - 5) ** 2 - (model.x1 - 6) ** 2 + 82.81) ** 2)
        else:
            obj4 = (-aml.log(-(model.x2 - 5) ** 2 - (model.x1 - 6) ** 2 + 82.81 + 1))
        if (100 - value(model.x1) + 1 <= 0.0):
            obj5 = (1E10 * (100 - model.x1) ** 2)
        else:
            obj5 = (-aml.log(100 - model.x1 + 1))
        if (value(model.x1) - 13 + 1 <= 0.0):
            obj6 = (1E10 * (model.x1 - 13) ** 2)
        else:
            obj6 = (-aml.log(model.x1 - 13 + 1))
        if (100 - value(model.x2) + 1 <= 0.0):
            obj7 = (1E10 * (100 - model.x2) ** 2)
        else:
            obj7 = (-aml.log(100 - model.x2 + 1))
        if (value(model.x2) + 1 <= 0.0):
            obj8 = (1E10 * (model.x2) ** 2)
        else:
            obj8 = (-aml.log(model.x2 + 1))
        return (model.x1 - 10) ** 3 + (model.x2 - 20) ** 3 + obj1 + obj2 + obj3 + obj4 + obj5 + obj6 + obj7 + obj8

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model110():

    # dqdrtic QUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dqdrtic'

    model.N = aml.Param(initialize=5000)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, initialize=3.0)

    model.SS = aml.RangeSet(1, model.N - 2)

    def f(model):
        return sum([(100 * model.x[i + 1] ** 2 + 100 * model.x[i + 2] ** 2 + model.x[i] ** 2) for i in model.SS])

    model.f = aml.Objective(rule=f)
    return model


def create_model111():

    # dqrtic OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'dqrtic'

    model.N = aml.Param(initialize=5000)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, initialize=2.0)

    def f(model):
        return sum([(model.x[i] - i) ** 4 for i in model.S])

    model.f = aml.Objective(rule=f)
    return model


# ToDo: try with line search
def create_model112():

    # drcav3lq NQR2-MY-V-V
    model = aml.ConcreteModel()
    model._name = 'drcav3lq'
    M = 100
    model.M = aml.Param(initialize=M)
    model.H = aml.Param(initialize=1.0 / (M + 2.0))
    model.RE = aml.Param(initialize=4500.0)

    model.S1 = aml.RangeSet(-1, M + 2)
    model.S2 = aml.RangeSet(1, M)

    model.y = aml.Var(model.S1, model.S1, initialize=0.0)

    def f(model):
        return 0

    model.f = aml.Objective(rule=f)

    def cons(model, i, j):
        return (20 * model.y[i, j] - 8 * model.y[i - 1, j] - 8 * model.y[i + 1, j] \
                - 8 * model.y[i, j - 1] - 8 * model.y[i, j + 1] + 2 * model.y[i - 1, j + 1] + 2 * model.y[
                    i + 1, j - 1] + 2 * model.y[i - 1, j - 1] + 2 * model.y[i + 1, j + 1] + \
                model.y[i - 2, j] + model.y[i + 2, j] + model.y[i, j - 2] + model.y[i, j + 2] + (model.RE / 4.0) * (
                model.y[i, j + 1] - model.y[i, j - 1]) \
                * (
                model.y[i - 2, j] + model.y[i - 1, j - 1] + model.y[i - 1, j + 1] - 4 * model.y[i - 1, j] - 4 * model.y[
                    i + 1, j] - model.y[i + 1, j - 1] \
                - model.y[i + 1, j + 1] - model.y[i + 2, j]) - (model.RE / 4.0) * (
                model.y[i + 1, j] - model.y[i - 1, j]) * \
                (
                model.y[i, j - 2] + model.y[i - 1, j - 1] + model.y[i + 1, j - 1] - 4 * model.y[i, j - 1] - 4 * model.y[
                    i, j + 1] - model.y[i - 1, j + 1] - model.y[i + 1, j + 1] - model.y[i, j + 2])) == 0

    model.cons = aml.Constraint(model.S2, model.S2, rule=cons)

    value = aml.value
    for j in model.S1:
        model.y[-1, j] = 0.0
        model.y[-1, j].fixed = True

        model.y[0, j] = 0.0
        model.y[0, j].fixed = True

    for i in model.S2:
        model.y[i, -1] = 0.0
        model.y[i, -1].fixed = True

        model.y[i, 0] = 0.0
        model.y[i, 0].fixed = True

        model.y[i, value(model.M) + 1] = 0.0
        model.y[i, value(model.M) + 1].fixed = True

        model.y[i, value(model.M) + 2] = 0.0
        model.y[i, value(model.M) + 2].fixed = True

    for j in model.S1:
        model.y[value(model.M) + 1, j] = -value(model.H) / 2.0
        model.y[value(model.M) + 1, j].fixed = True

        model.y[value(model.M) + 2, j] = value(model.H) / 2.0
        model.y[value(model.M) + 2, j].fixed = True

    return model


def create_model113():

    # dtoc1l OLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'dtoc1l'

    n = 1000
    nx = 5
    ny = 10

    RangeSet = aml.RangeSet
    model.S1 = RangeSet(1, ny)
    model.S2 = RangeSet(1, nx)

    def b_rule(model, i, j):
        return float(i - j) / float(nx + ny)

    model.b = aml.Param(model.S1, model.S2, initialize=b_rule)

    model.S3 = RangeSet(1, n - 1)
    model.S4 = RangeSet(1, n)
    model.S5 = RangeSet(2, ny - 1)

    model.x = aml.Var(model.S3, model.S2)
    model.y = aml.Var(model.S4, model.S1)

    def f(model):
        sum1 = sum((model.x[t, i] + 0.5) ** 4 for t in model.S3 for i in model.S2)
        sum2 = sum((model.y[t, i] + 0.25) ** 4 for t in model.S4 for i in model.S1)
        return sum1 + sum2

    model.f = aml.Objective(rule=f)

    def cons1(model, t):
        sc1 = sum(aml.value(model.b[1, i]) * model.x[t, i] for i in model.S2)
        return 0.5 * model.y[t, 1] + 0.25 * model.y[t, 2] - model.y[t + 1, 1] + sc1 == 0

    model.cons1 = aml.Constraint(model.S3, rule=cons1)

    def cons2(model, t, j):
        sc2 = sum(aml.value(model.b[j, i]) * model.x[t, i] for i in model.S2)
        return -model.y[t + 1, j] + 0.5 * model.y[t, j] - 0.25 * model.y[t, j - 1] + 0.25 * model.y[t, j + 1] + sc2 == 0

    model.cons2 = aml.Constraint(model.S3, model.S5, rule=cons2)

    def cons3(model, t):
        sc3 = sum(model.b[ny, i] * model.x[t, i] for i in model.S2)
        return 0.5 * model.y[t, ny] - 0.25 * model.y[t, ny - 1] - model.y[t + 1, ny] + sc3 == 0

    model.cons3 = aml.Constraint(model.S3, rule=cons3)

    for idx in model.S1:
        model.y[1, idx] = 0.0
        model.y[1, idx].fixed = True

    return model


def create_model114():

    # edensch OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'edensch'

    N = 2000
    model.x = aml.Var(aml.RangeSet(1, N), initialize=0.0)

    def f_rule(model):
        return sum((model.x[i] - 2) ** 4 + (model.x[i] * model.x[i + 1] - 2 * model.x[i + 1]) ** 2 + \
                   (model.x[i + 1] + 1) ** 2 for i in range(1, N)) + 16

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model115():

    # eg1 OBR2-AY-3-0
    model = aml.ConcreteModel()
    model._name = 'eg1'
    model.x1 = aml.Var()
    model.x2 = aml.Var(bounds=(-1.0, 1.0))
    model.x3 = aml.Var(bounds=(1.0, 2.0))

    def obj(m):
        return (m.x1 ** 2 + (m.x2 * m.x3) ** 4 + m.x1 * m.x3 + m.x2 * aml.sin(m.x1 + m.x3) + m.x2)

    model.obj = aml.Objective(rule=obj)
    return model


def create_model116():

    # eg2 OBR2-AY-3-0
    model = aml.ConcreteModel()
    model._name = 'eg2'
    model.N = aml.Param(initialize=1000)
    model.S = aml.RangeSet(1, model.N)

    model.x = aml.Var(model.S)

    def f(m):
        return sum(aml.sin(m.x[1] + m.x[i] ** 2 - 1.0) for i in range(1, aml.value(m.N))) + \
               0.5 * aml.sin(m.x[aml.value(m.N)] ** 2)

    model.f = aml.Objective(rule=f)

    return model


def create_model117():

    # eg3 OOR2-AY-V-V
    model = aml.ConcreteModel()
    model._name = 'eg3'

    model.n = aml.Param(initialize=100)
    model.S = aml.RangeSet(1, model.n)

    model.y = aml.Var()

    def bounds(model, i):
        return (-1, i)

    model.x = aml.Var(model.S, initialize=0.5, bounds=bounds)

    def f(m):
        expr = 0.5 * ((m.x[1] - m.x[aml.value(m.n)]) * m.x[2] + m.y) ** 2
        return expr

    model.f = aml.Objective(rule=f)

    def consq(m, i):
        expr = m.y + m.x[1] * m.x[i + 1] + (1 + float(2) / float(i)) * m.x[i] * m.x[aml.value(m.n)]
        return (None, expr, 0.0)

    model.consq = aml.Constraint(aml.RangeSet(1, model.n - 1), rule=consq)

    def conss(m, i):
        expr = (aml.sin(m.x[i])) ** 2
        return (None, expr, 0.5)

    model.conss = aml.Constraint(model.S, rule=conss)

    def eq(m):
        return (1.0, (m.x[1] + m.x[aml.value(m.n)]) ** 2)

    model.eq = aml.Constraint(rule=eq)

    return model


def create_model118():

    # eigenval OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'eigenval'

    model.N = aml.Param(initialize=5000)
    model.S1 = aml.RangeSet(1, model.N)
    model.S2 = aml.RangeSet(1, model.N - 1)
    model.x = aml.Var(model.S1, initialize=2.0)

    def f(model):
        sumexp1 = 0
        sumexp2 = 0
        for i in model.S2:
            sumexp1 += -4 * model.x[i] + 3.0
            sumexp2 += (model.x[i] ** 2 + model.x[i + 1] ** 2) ** 2
        return sumexp1 + sumexp2

    model.f = aml.Objective(rule=f)
    return model


def create_model119():

    # eigenval2 SUR2-AN-3-0
    model = aml.ConcreteModel()
    model._name = 'eigenval2'

    x_init = dict()
    x_init[1] = 1.0
    x_init[2] = 2.0
    x_init[3] = 0.0

    model.x = aml.Var(aml.RangeSet(1, 3), initialize=x_init)

    def f_rule(model):
        return (model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 - 1) ** 2 \
               + (model.x[1] ** 2 + model.x[2] ** 2 + (model.x[3] - 2) ** 2 - 1) ** 2 \
               + (model.x[1] + model.x[2] + model.x[3] - 1) ** 2 \
               + (model.x[1] + model.x[2] - model.x[3] + 1) ** 2 \
               + (3 * model.x[2] ** 2 + model.x[1] ** 3 + (5 * model.x[3] - model.x[1] + 1) ** 2 - 36) ** 2

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model120():

    # expfitc OLR2-AN-5-502
    model = aml.ConcreteModel()
    model._name = 'expfitc'
    Param = aml.Param
    RangeSet = aml.RangeSet

    R = 251.0

    def T_rule(model, i):
        return 5 * (i - 1) / (R - 1)

    model.T = Param(RangeSet(1, R), initialize=T_rule, mutable=True)

    def ET_rule(model, i):
        return aml.exp(model.T[i])

    model.ET = Param(RangeSet(1, R), initialize=ET_rule)

    pinit = dict()
    pinit[0] = 1.0
    pinit[1] = 1.0
    pinit[2] = 6.0

    model.P = aml.Var(RangeSet(0, 2), initialize=pinit)

    model.Q = aml.Var(RangeSet(1, 2), initialize=0.0)

    def f_rule(model):
        return sum(( \
                       (model.P[0] + model.P[1] * model.T[i] + model.P[2] * model.T[i] ** 2) / \
                       (model.ET[i] * (1 + model.Q[1] * (model.T[i] - 5) + model.Q[2] * (model.T[i] - 5) ** 2)) \
                       - 1) ** 2 for i in range(1, int(R) + 1))

    model.f = aml.Objective(rule=f_rule)

    def cons1(model, i):
        return model.P[0] + model.P[1] * model.T[i] + model.P[2] * model.T[i] ** 2 - (model.T[i] - 5) * model.ET[i] * \
                                                                                     model.Q[1] - \
               (model.T[i] - 5) ** 2 * model.ET[i] * model.Q[2] - model.ET[i] >= 0

    model.cons1 = aml.Constraint(RangeSet(1, R), rule=cons1)

    def cons2(model, i):
        return (model.T[i] - 5) * model.Q[1] + (model.T[i] - 5) ** 2 * model.Q[2] + 0.99999 >= 0

    model.cons2 = aml.Constraint(RangeSet(1, R), rule=cons2)

    return model


def create_model121():

    # expfitb OLR2-AN-5-102
    model = aml.ConcreteModel()
    model._name = 'expfitb'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    Constraint = aml.Constraint
    R = 51.0

    def T_rule(model, i):
        return 5 * (i - 1) / (R - 1)

    model.T = Param(RangeSet(1, R), initialize=T_rule, mutable=True)

    def ET_rule(model, i):
        return aml.exp(model.T[i])

    model.ET = Param(RangeSet(1, R), initialize=ET_rule)

    pinit = dict()
    pinit[0] = 1.0
    pinit[1] = 1.0
    pinit[2] = 6.0

    model.P = Var(RangeSet(0, 2), initialize=pinit)

    model.Q = Var(RangeSet(1, 2), initialize=0.0)

    # For Pyomo testing,
    # generate the ConcreteModel version
    # by loading the data
    import os
    if os.path.isfile(os.path.abspath(__file__).replace('.pyc', '.dat').replace('.py', '.dat')):
        model = model.create_instance(os.path.abspath(__file__).replace('.pyc', '.dat').replace('.py', '.dat'))

    def f_rule(model):
        return sum(( \
                       (model.P[0] + model.P[1] * model.T[i] + model.P[2] * model.T[i] ** 2) / \
                       (model.ET[i] * (1 + model.Q[1] * (model.T[i] - 5) + model.Q[2] * (model.T[i] - 5) ** 2)) \
                       - 1) ** 2 for i in range(1, int(R) + 1))

    model.f = aml.Objective(rule=f_rule)

    def cons1(model, i):
        return model.P[0] + model.P[1] * model.T[i] + model.P[2] * model.T[i] ** 2 - (model.T[i] - 5) * model.ET[i] * \
                                                                                     model.Q[1] - \
               (model.T[i] - 5) ** 2 * model.ET[i] * model.Q[2] - model.ET[i] >= 0

    model.cons1 = Constraint(RangeSet(1, R), rule=cons1)

    def cons2(model, i):
        return (model.T[i] - 5) * model.Q[1] + (model.T[i] - 5) ** 2 * model.Q[2] + 0.99999 >= 0

    model.cons2 = Constraint(RangeSet(1, R), rule=cons2)

    return model


def create_model122():

    # expfit SUR2-AN-2-0
    model = aml.ConcreteModel()
    model._name = 'expfit'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    Constraint = aml.Constraint

    model.p = Param(initialize=10)
    model.h = Param(initialize=0.25)
    model.alpha = Var()
    model.beta = Var()
    model.S = RangeSet(1, model.p)

    def f(model):
        return sum([(model.alpha * aml.exp(i * aml.value(model.h) * model.beta) - i * aml.value(model.h)) ** 2 for i in model.S])

    model.f = aml.Objective(rule=f)

    return model


def create_model123():

    # epfit1 OLR2-AN-5-22
    model = aml.ConcreteModel()
    model._name = 'epfit1'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    Constraint = aml.Constraint

    R = 11.0

    def T_rule(model, i):
        return 5 * (i - 1) / (R - 1)

    model.T = Param(RangeSet(1, R), initialize=T_rule, mutable=True)

    def ET_rule(model, i):
        return aml.exp(model.T[i])

    model.ET = Param(RangeSet(1, R), initialize=ET_rule)

    pinit = dict()
    pinit[0] = 1.0
    pinit[1] = 1.0
    pinit[2] = 6.0

    model.P = Var(RangeSet(0, 2), initialize=pinit)

    model.Q = Var(RangeSet(1, 2), initialize=0.0)

    # For Pyomo testing,
    # generate the ConcreteModel version
    # by loading the data
    import os
    if os.path.isfile(os.path.abspath(__file__).replace('.pyc', '.dat').replace('.py', '.dat')):
        model = model.create_instance(os.path.abspath(__file__).replace('.pyc', '.dat').replace('.py', '.dat'))

    def f_rule(model):
        return sum(( \
                       (model.P[0] + model.P[1] * model.T[i] + model.P[2] * model.T[i] ** 2) / \
                       (model.ET[i] * (1 + model.Q[1] * (model.T[i] - 5) + model.Q[2] * (model.T[i] - 5) ** 2)) \
                       - 1) ** 2 for i in range(1, int(R) + 1))

    model.f = aml.Objective(rule=f_rule)

    def cons1(model, i):
        return model.P[0] + model.P[1] * model.T[i] + model.P[2] * model.T[i] ** 2 - (model.T[i] - 5) * model.ET[i] * \
                                                                                     model.Q[1] - \
               (model.T[i] - 5) ** 2 * model.ET[i] * model.Q[2] - model.ET[i] >= 0

    model.cons1 = Constraint(RangeSet(1, R), rule=cons1)

    def cons2(model, i):
        return (model.T[i] - 5) * model.Q[1] + (model.T[i] - 5) ** 2 * model.Q[2] + 0.99999 >= 0

    model.cons2 = Constraint(RangeSet(1, R), rule=cons2)
    return model


def create_model124():

    # errinros SUR2-AN-V-0

    model = aml.ConcreteModel()
    model._name = 'errinros'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    Constraint = aml.Constraint

    model.N = Param(initialize=50)
    model.S = RangeSet(1, model.N)
    alpha = dict()
    alpha[1] = 1.25
    alpha[2] = 1.40
    alpha[3] = 2.40
    alpha[4] = 1.40
    alpha[5] = 1.75
    alpha[6] = 1.20
    alpha[7] = 2.25
    alpha[8] = 1.20
    alpha[9] = 1.00
    alpha[10] = 1.10
    alpha[11] = 1.50
    alpha[12] = 1.60
    alpha[13] = 1.25
    alpha[14] = 1.25
    alpha[15] = 1.20
    alpha[16] = 1.20
    alpha[17] = 1.40
    alpha[18] = 0.50
    alpha[19] = 0.50
    alpha[20] = 1.25
    alpha[21] = 1.80
    alpha[22] = 0.75
    alpha[23] = 1.25
    alpha[24] = 1.40
    alpha[25] = 1.60
    alpha[26] = 2.00
    alpha[27] = 1.00
    alpha[28] = 1.60
    alpha[29] = 1.25
    alpha[30] = 2.75
    alpha[31] = 1.25
    alpha[32] = 1.25
    alpha[33] = 1.25
    alpha[34] = 3.00
    alpha[35] = 1.50
    alpha[36] = 2.00
    alpha[37] = 1.25
    alpha[38] = 1.40
    alpha[39] = 1.80
    alpha[40] = 1.50
    alpha[41] = 2.20
    alpha[42] = 1.40
    alpha[43] = 1.50
    alpha[44] = 1.25
    alpha[45] = 2.00
    alpha[46] = 1.50
    alpha[47] = 1.25
    alpha[48] = 1.40
    alpha[49] = 0.60
    alpha[50] = 1.50
    model.x = Var(model.S, initialize=-1.0)

    model.SS = RangeSet(2, model.N)

    def f(model):
        sum1 = sum([(model.x[i - 1] - 16 * alpha[i] ** 2 * model.x[i] ** 2) ** 2 for i in model.SS])
        sum2 = sum([(model.x[i] - 1.0) ** 2 for i in model.SS])
        return sum1 + sum2

    model.f = aml.Objective(rule=f)
    return model


# ToDo: works but try with linesearch
def create_model125():

    # explin2 OBR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'explin2'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    Constraint = aml.Constraint

    model.n = Param(initialize=120)
    model.m = Param(initialize=10)

    model.S = RangeSet(1, model.n)
    model.x = Var(model.S, bounds=(0, 10.0), initialize=0.0)

    model.SS = RangeSet(1, model.m)

    def f(model):
        sum1 = sum([aml.exp(0.1 * i * model.x[i] * model.x[i + 1] / aml.value(model.m)) for i in model.SS])
        sum2 = sum([(-10.0 * i * model.x[i]) for i in model.S])
        return sum1 + sum2

    model.f = aml.Objective(rule=f)
    return model


def create_model126():
    # explin OBR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'explin'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var

    model.n = Param(initialize=120)
    model.m = Param(initialize=10)

    model.S = RangeSet(1, model.n)
    model.x = Var(model.S, bounds=(0, 10.0), initialize=0.0)

    model.SS = RangeSet(1, model.m)

    def f(model):
        sum1 = sum([aml.exp(0.1 * model.x[i] * model.x[i + 1]) for i in model.SS])
        sum2 = sum([(-10.0 * i * model.x[i]) for i in model.S])
        return sum1 + sum2

    model.f = aml.Objective(rule=f)

    return model


def create_model127():

    # expquad OBR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'expquad'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var

    model.n = Param(initialize=120)
    model.m = Param(initialize=10)

    model.S = RangeSet(1, model.n)
    model.x = Var(model.S, initialize=0.0)

    model.SS = RangeSet(1, model.m)
    model.SSS = RangeSet(model.m + 1, model.n - 1)

    def f(model):
        sum1 = sum(aml.exp(0.1 * i * aml.value(model.m) * model.x[i] * model.x[i + 1]) for i in model.SS)
        sum2 = sum((4.0 * model.x[i] * model.x[i] + 2.0 * model.x[aml.value(model.n)] * model.x[aml.value(model.n)] + model.x[
            i] * model.x[aml.value(model.n)]) for i in model.SSS)
        sum3 = sum((-10.0 * i * model.x[i]) for i in model.S)
        return sum1 + sum2 + sum3

    model.f = aml.Objective(rule=f)

    def cons(model, i):
        return (0.0, model.x[i], 10.0)

    model.cons = aml.Constraint(model.SS, rule=cons)

    return model


def create_model128():

    # extrasim LLR2-AN-2-1
    model = aml.ConcreteModel()
    model._name = 'extrasim'

    model.x = aml.Var(bounds=(0, None))
    model.y = aml.Var()

    def f(model):
        return model.x + 1

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.x + 2 * model.y - 2.0 == 0

    model.cons1 = aml.Constraint(rule=cons1)
    return model


def create_model129():

    # extrosnb SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'extrosnb'

    model.N = aml.Param(initialize=10)
    model.S = aml.RangeSet(1, model.N)
    model.x = aml.Var(model.S, initialize=1)

    model.Sf = aml.RangeSet(2, model.N)

    def f(model):
        sumexp = sum([100 * (model.x[i] - model.x[i - 1] ** 2) ** 2 for i in model.Sf])
        return (model.x[1] - 1) ** 2 + sumexp

    model.f = aml.Objective(rule=f)
    return model


def create_model130():

    # fccu SLR2-MN-19-8
    w = dict()
    w[1] = 0.2
    w[2] = 1
    w[3] = 1
    w[4] = 0.33333333
    w[5] = 0.33333333
    w[6] = 0.33333333
    w[7] = 1
    w[8] = 1
    w[9] = 1
    w[10] = 1
    w[11] = 1
    w[12] = 1
    w[13] = 1
    w[14] = 1
    w[15] = 0.33333333
    w[16] = 0.33333333
    w[17] = 1
    w[18] = 0.33333333
    w[19] = 0.33333333

    m = dict()
    m[1] = 31
    m[2] = 36
    m[3] = 20
    m[4] = 3
    m[5] = 5
    m[6] = 3.5
    m[7] = 4.2
    m[8] = 0.9
    m[9] = 3.9
    m[10] = 2.2
    m[11] = 22.8
    m[12] = 6.8
    m[13] = 19
    m[14] = 8.5
    m[15] = 2.2
    m[16] = 2.5
    m[17] = 10.8
    m[18] = 6.5
    m[19] = 6.5

    model = aml.ConcreteModel()

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.S = RangeSet(1, 19)
    model._name = 'fccu'
    model.w = w
    model.m = m

    model.Feed = Var(initialize=1)
    model.Effluent = Var(initialize=1)
    model.MF_ohd = Var(initialize=1)
    model.HCN = Var(initialize=1)
    model.LCO = Var(initialize=1)
    model.HCO = Var(initialize=1)
    model.MF_btms = Var(initialize=1)
    model.Decant = Var(initialize=1)
    model.Dec_recy = Var(initialize=1)
    model.Off_gas = Var(initialize=1)
    model.DC4_feed = Var(initialize=1)
    model.DC3_feed = Var(initialize=1)
    model.DC4_btms = Var(initialize=1)
    model.Lean_oil = Var(initialize=1)
    model.Propane = Var(initialize=1)
    model.Butane = Var(initialize=1)
    model.C8spl_fd = Var(initialize=1)
    model.LCN = Var(initialize=1)
    model.MCN = Var(initialize=1)

    def f(model):
        return (model.Feed - value(model.m[1])) ** 2 / value(model.w[1]) \
               + (model.Effluent - value(model.m[2])) ** 2 / value(model.w[2]) \
               + (model.MF_ohd - value(model.m[3])) ** 2 / value(model.w[3]) \
               + (model.HCN - value(model.m[4])) ** 2 / value(model.w[4]) \
               + (model.LCO - value(model.m[5])) ** 2 / value(model.w[5]) \
               + (model.HCO - value(model.m[6])) ** 2 / value(model.w[6]) \
               + (model.MF_btms - value(model.m[7])) ** 2 / value(model.w[7]) \
               + (model.Decant - value(model.m[8])) ** 2 / value(model.w[8]) \
               + (model.Dec_recy - value(model.m[9])) ** 2 / value(model.w[9]) \
               + (model.Off_gas - value(model.m[10])) ** 2 / value(model.w[10]) \
               + (model.DC4_feed - value(model.m[11])) ** 2 / value(model.w[11]) \
               + (model.DC3_feed - value(model.m[12])) ** 2 / value(model.w[12]) \
               + (model.DC4_btms - value(model.m[13])) ** 2 / value(model.w[13]) \
               + (model.Lean_oil - value(model.m[14])) ** 2 / value(model.w[14]) \
               + (model.Propane - value(model.m[15])) ** 2 / value(model.w[15]) \
               + (model.Butane - value(model.m[16])) ** 2 / value(model.w[16]) \
               + (model.C8spl_fd - value(model.m[17])) ** 2 / value(model.w[17]) \
               + (model.LCN - value(model.m[18])) ** 2 / value(model.w[18]) \
               + (model.MCN - value(model.m[19])) ** 2 / value(model.w[19])

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return model.Feed + model.Dec_recy - model.Effluent == 0

    model.cons1 = Constraint(rule=cons1)

    def cons2(model):
        return model.Effluent - model.MF_ohd - model.HCN - model.LCO - model.HCO - model.MF_btms == 0

    model.cons2 = Constraint(rule=cons2)

    def cons3(model):
        return model.MF_btms - model.Decant - model.Dec_recy == 0

    model.cons3 = Constraint(rule=cons3)

    def cons4(model):
        return model.MF_ohd + model.Lean_oil - model.Off_gas - model.DC4_feed == 0

    model.cons4 = Constraint(rule=cons4)

    def cons5(model):
        return model.DC4_feed - model.DC3_feed - model.DC4_btms == 0

    model.cons5 = Constraint(rule=cons5)

    def cons6(model):
        return model.DC4_btms - model.Lean_oil - model.C8spl_fd == 0

    model.cons6 = Constraint(rule=cons6)

    def cons7(model):
        return model.DC3_feed - model.Propane - model.Butane == 0

    model.cons7 = Constraint(rule=cons7)

    def cons8(model):
        return model.C8spl_fd - model.LCN - model.MCN == 0

    model.cons8 = Constraint(rule=cons8)
    return model


def create_model131():

    # fletchbv OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'fletchby'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value

    model.n = Param(initialize=10)
    model.kappa = Param(initialize=1.0)
    model.objscale = Param(initialize=1.0e0)

    def h_rule(model):
        return 1 / (value(model.n) + 1.0)

    model.h = Param(initialize=h_rule)

    def p_rule(model):
        return 1 / value(model.objscale)

    model.p = Param(initialize=p_rule)

    model.S = RangeSet(1, model.n)
    model.SS = RangeSet(1, model.n - 1)

    def x_int(model, i):
        return i * value(model.h)

    model.x = Var(model.S, initialize=x_int)

    def f(model):
        exp1 = 0.5 * value(model.p) * (model.x[1]) ** 2
        exp2 = 0.5 * value(model.p) * (model.x[value(model.n)]) ** 2
        sum1 = sum([0.5 * value(model.p) * (model.x[i] - model.x[i + 1]) ** 2 for i in model.SS])
        sum2 = sum([(value(model.p) * (-1 - 2 / value(model.h) ** 2) * model.x[i]) for i in model.S])
        sum3 = sum([(-value(model.kappa) * value(model.p) * aml.cos(model.x[i]) / value(model.h) ** 2) for i in model.S])
        return (exp1 + sum1 + exp2 + sum2 + sum3)

    model.f = aml.Objective(rule=f)
    return model


def create_model132():

    # fletchcr OUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'fletchcr'

    N = 100

    model.x = aml.Var(aml.RangeSet(1, N), initialize=0.0)

    def f_rule(model):
        return sum(100 * (model.x[i + 1] - model.x[i] + 1 - model.x[i] ** 2) ** 2 for i in range(1, N))

    model.f = aml.Objective(rule=f_rule)
    return model


# ToDo: this fail in first iteration
def create_model133():

    # fletcher QOR2-AN-4-4
    model = aml.ConcreteModel()
    model._name = 'fletcher'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.S1 = RangeSet(1, 4)
    model.x = Var(model.S1, initialize=10.0)

    def f(model):
        return model.x[1] * model.x[2]

    model.f = aml.Objective(rule=f)

    def cons1(model):
        return (model.x[1] * model.x[3] + model.x[2] * model.x[4]) ** 2 / (model.x[1] ** 2 + model.x[2] ** 2) - model.x[
                                                                                                                    3] ** 2 - \
               model.x[4] ** 2 + 1 == 0

    model.cons1 = Constraint(rule=cons1)

    def cons2(model):
        return model.x[1] - model.x[3] - 1 >= 0

    model.cons2 = Constraint(rule=cons2)

    def cons3(model):
        return model.x[2] - model.x[4] - 1 >= 0

    model.cons3 = Constraint(rule=cons3)

    def cons4(model):
        return model.x[3] - model.x[4] >= 0

    model.cons4 = Constraint(rule=cons4)

    def cons5(model):
        return model.x[4] >= 1

    model.cons5 = Constraint(rule=cons5)
    return model


def create_model134():

    # fminsrf2 OUR2-MY-V-0
    model = aml.ConcreteModel()
    model._name = 'fminsrf2'

    p = 32

    h00 = 1.0
    slopej = 4.0
    slopei = 8.0

    scale = (p - 1) ** 2

    ston = slopei / (p - 1)
    wtoe = slopej / (p - 1)
    h01 = h00 + slopej
    h10 = h00 + slopei
    mid = p / 2

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p))

    def f_rule(model):
        return sum(((0.5 * (p - 1) ** 2 * (
        (model.x[i, j] - model.x[i + 1, j + 1]) ** 2 + (model.x[i + 1, j] - model.x[i, j + 1]) ** 2) + 1.0 \
                     )) ** 0.5 for i in range(1, p) for j in range(1, p)) / scale + \
               (model.x[mid, mid]) ** 2 / p ** 2

    model.f = aml.Objective(rule=f_rule)

    for j in range(1, p + 1):
        model.x[1, j] = (j - 1) * wtoe + h00

    for j in range(1, p + 1):
        model.x[p, j] = (j - 1) * wtoe + h10

    for i in range(2, p):
        model.x[i, p] = (i - 1) * ston + h00

    for i in range(2, p):
        model.x[i, 1] = (i - 1) * ston + h01

    for i in range(2, p):
        for j in range(2, p):
            model.x[i, j] = 0.0

    return model


def create_model135():

    # fminsurf OUR2-MY-V-0
    model = aml.ConcreteModel()
    model._name = 'fminsurf'

    p = 32

    h00 = 1.0
    slopej = 4.0
    slopei = 8.0

    scale = (p - 1) ** 2

    ston = slopei / (p - 1)
    wtoe = slopej / (p - 1)
    h01 = h00 + slopej
    h10 = h00 + slopei

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p))

    def f_rule(model):
        return sum((0.5 * (p - 1) ** 2 * (
        (model.x[i, j] - model.x[i + 1, j + 1]) ** 2 + (model.x[i + 1, j] - model.x[i, j + 1]) ** 2) + 1.0) \
                   ** 0.5 for i in range(1, p) for j in range(1, p)) / scale + \
               (sum(model.x[i, j] for j in range(1, p + 1) for i in range(1, p + 1))) ** 2 / p ** 4

    model.f = aml.Objective(rule=f_rule)

    for j in range(1, p + 1):
        model.x[1, j] = (j - 1) * wtoe + h00

    for j in range(1, p + 1):
        model.x[p, j] = (j - 1) * wtoe + h10

    for i in range(2, p):
        model.x[i, p] = (i - 1) * ston + h00

    for i in range(2, p):
        model.x[i, 1] = (i - 1) * ston + h01

    for i in range(2, p):
        for j in range(2, p):
            model.x[i, j] = 0.0

    return model


def create_model136():

    # freuroth SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'freuroth'

    n = 5000
    ngs = n - 1

    model.x = aml.Var(aml.RangeSet(1, n))

    def f_rule(model):
        return sum(((5.0 - model.x[i + 1]) * model.x[i + 1] ** 2 + model.x[i] - 2 * model.x[i + 1] - 13.0) ** 2 for i in
                   range(1, ngs + 1)) + \
               sum(((1.0 + model.x[i + 1]) * model.x[i + 1] ** 2 + model.x[i] - 14 * model.x[i + 1] - 29.0) ** 2 for i
                   in range(1, ngs + 1))

    model.f = aml.Objective(rule=f_rule)

    model.x[1] = 0.5
    model.x[2] = -2.0

    for i in range(3, n + 1):
        model.x[i] = 0.0

    return model


# ToDo: needs more memory
def create_model137():

    # gausselm LOR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'gausselm'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    n = 16

    def x_init_rule(model, k, i, j):
        if i == j:
            return 1.0
        else:
            return 0.01

    model.x = Var([(k, i, j) for k in range(1, n + 1) for i in range(k, n + 1) for j in range(k, n + 1)],
                  initialize=x_init_rule)

    def f_rule(model):
        return -model.x[n, n, n]

    model.f = aml.Objective(rule=f_rule)

    conse_index = [(k, i, j) for k in range(1, n - 1 + 1) for i in range(k + 1, n + 1) for j in range(k + 1, n + 1)]

    def conse_rule(model, k, i, j):
        return model.x[k, i, k] * model.x[k, k, j] / model.x[k, k, k] + model.x[k + 1, i, j] - model.x[k, i, j] == 0

    model.conse = Constraint(conse_index, rule=conse_rule)

    consmikk_index = [(k, i) for k in range(2, n) for i in range(k + 1, n + 1)]

    def consmikk_rule(model, k, i):
        return model.x[k, i, k] - model.x[k, k, k] <= 0

    model.consmikk = Constraint(consmikk_index, rule=consmikk_rule)

    def consmkik_rule(model, k, i):
        return model.x[k, k, i] - model.x[k, k, k] <= 0

    model.consmkik = Constraint(consmikk_index, rule=consmkik_rule)

    consmijk_index = [(k, i, j) for k in range(2, n) for i in range(k + 1, n + 1) for j in range(k + 1, n + 1)]

    def consmijk_rule(model, k, i, j):
        return model.x[k, i, j] - model.x[k, k, k] <= 0

    model.consmijk = Constraint(consmijk_index, rule=consmijk_rule)

    def conspikk_rule(model, k, i):
        return model.x[k, i, k] + model.x[k, k, k] >= 0

    model.conspikk = Constraint(consmikk_index, rule=conspikk_rule)

    def conspkik_rule(model, k, i):
        return model.x[k, k, i] + model.x[k, k, k] >= 0

    model.conspkik = Constraint(consmikk_index, rule=conspkik_rule)

    def conspijk_rule(model, k, i, j):
        return model.x[k, i, j] + model.x[k, k, k] >= 0

    model.conspijk = Constraint(consmijk_index, rule=conspijk_rule)

    def var_bnd_rule(model, i, j):
        return -1.0 <= model.x[1, i, j] <= 1.0

    model.var_bnd = Constraint(RangeSet(1, n), RangeSet(1, n), rule=var_bnd_rule)

    def var_bnd_diag_rule(model, k):
        return model.x[k, k, k] >= 0.0

    model.var_bnd_diag = Constraint(RangeSet(1, n), rule=var_bnd_diag_rule)

    model.x[1, 1, 1] = 1.0
    model.x[1, 1, 1].fixed = True

    return model


def create_model138():

    # genhs28 QLR2-AY-10-8
    model = aml.ConcreteModel()
    model._name = 'genhs28'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    N = 10

    def x_init_rule(model, i):
        if i == 1:
            return -4.0
        else:
            return 1.0

    model.x = Var(RangeSet(1, N), initialize=x_init_rule)

    def f_rule(model):
        return sum((model.x[i] + model.x[i + 1]) ** 2 for i in range(1, N))

    model.f = aml.Objective(rule=f_rule)

    def cons_rule(model, i):
        return -1.0 + model.x[i] + 2 * model.x[i + 1] + 3 * model.x[i + 2] == 0

    model.cons = Constraint(RangeSet(1, N - 2), rule=cons_rule)
    return model


# ToDo: pass but check with linesearch different conditions
def create_model139():

    # genhumps SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'genhumps'

    zeta = 2
    N = 5

    def x_init_rule(model, i):
        if i == 1:
            return -506.0
        else:
            return 506.2

    model.x = aml.Var(aml.RangeSet(1, N), initialize=x_init_rule)

    def f_rule(model):
        return sum(aml.sin(zeta * model.x[i]) ** 2 * aml.sin(zeta * model.x[i + 1]) ** 2 + 0.05 * (
        model.x[i] ** 2 + model.x[i + 1] ** 2) for i in range(1, N))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model140():

    # genrose SUR2-AN-V-0
    model = aml.ConcreteModel()
    model._name = 'genrose'

    n = 500

    model.x = aml.Var(aml.RangeSet(1, n), initialize=1.0 / (n + 1.0))

    def f_rule(model):
        return 1.0 + sum(100 * (model.x[i] - model.x[i - 1] ** 2) ** 2 for i in range(2, n + 1)) + \
               sum((model.x[i] - 1.0) ** 2 for i in range(2, n + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model141():

    # gigomez1 LQR2-AN-3-3
    model = aml.ConcreteModel()
    model._name = 'gigomez1'

    model.x = aml.Var([1, 2], initialize=2.0)
    model.z = aml.Var(initialize=2.0)

    model.f = aml.Objective(expr=model.z)

    model.cons1 = aml.Constraint(expr=model.z + 5.0 * model.x[1] - model.x[2] >= 0)
    model.cons2 = aml.Constraint(expr=model.z - 4.0 * model.x[2] - model.x[1] ** 2 - model.x[2] ** 2 >= 0)
    model.cons3 = aml.Constraint(expr=model.z - 5.0 * model.x[1] - model.x[2] >= 0)
    return model


def create_model142():

    # gilbert QQR2-AN-V-1
    model = aml.ConcreteModel()
    model._name = 'gilbert'

    n = 1000

    def x_init_rule(model, i):
        return (-1.0) ** (i + 1) * 10.0

    model.x = aml.Var(aml.RangeSet(1, n), initialize=x_init_rule)

    def f_rule(model):
        return sum(((n + 1 - i) * model.x[i] / n - 1.0) ** 2 for i in range(1, n + 1)) / 2.0

    model.f = aml.Objective(rule=f_rule)

    def cons1_rule(model):
        return (sum(model.x[i] ** 2 for i in range(1, n + 1)) - 1.0) / 2.0 == 0.0

    model.cons1 = aml.Constraint(rule=cons1_rule)

    return model


def create_model143():

    # goffin LLR2-AN-51-50
    model = aml.ConcreteModel()
    model._name = 'goffin'

    ri = 50
    t = -25.5 + (50)

    def x_init_rule(model, j):
        return -25.5 + j

    model.x = aml.Var(aml.RangeSet(1, ri), initialize=x_init_rule)

    model.u = aml.Var()

    model.obj = aml.Objective(expr=model.u)

    def f_rule(model, i):
        return model.u >= 50 * model.x[i] - aml.summation(model.x)

    model.f = aml.Constraint(aml.RangeSet(1, ri), rule=f_rule)
    return model


def create_model144():

    # gottfr NQR2-AN-2-2
    model = aml.ConcreteModel()
    model._name = 'gottfr'

    model.x = aml.Var(aml.RangeSet(1, 2), initialize=0.5)

    def f_rule(model):
        return 0

    model.f = aml.Objective(rule=f_rule)

    def cons_rule(model):
        return (model.x[1] - 0.1136 * (model.x[1] + 3.0 * model.x[2]) * (1 - model.x[1])) == 0

    model.cons = aml.Constraint(rule=cons_rule)

    def cons2_rule(model):
        return (model.x[2] + 7.5 * (2.0 * model.x[1] - model.x[2]) * (1 - model.x[2])) == 0

    model.cons2 = aml.Constraint(rule=cons2_rule)

    return model


# ToDo: pass but try with linesearch
def create_model145():

    # gouldqp2 QLR2-MN-V-V
    model = aml.ConcreteModel()
    model._name = 'gouldqp2'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    K = 350

    def alpha_rule(model, i):
        if i > 1:
            return 1.0 + 1.01 ** i
        else:
            return 2.0

    model.alpha = Param(RangeSet(1, K + 1), initialize=alpha_rule)

    def knot_init_rule(model, i):
        return model.alpha[i]

    def knot_bounds_rule(model, i):
        return (model.alpha[i], model.alpha[i + 1])

    model.knot = Var(RangeSet(1, K), initialize=knot_init_rule, bounds=knot_bounds_rule)

    def space_init_rule(model, i):
        return model.alpha[i + 1] - model.alpha[i]

    def space_bounds_rule(model, i):
        return (0.4 * (model.alpha[i + 2] - model.alpha[i]), 0.6 * (model.alpha[i + 2] - model.alpha[i]))

    model.space = Var(RangeSet(1, K - 1), initialize=space_init_rule, bounds=space_bounds_rule)

    def f_rule(model):
        return sum(.5 * (model.space[i + 1] - model.space[i]) ** 2 for i in range(1, K - 1))

    model.f = aml.Objective(rule=f_rule)

    def cons1_rule(model, i):
        return model.space[i] - model.knot[i + 1] + model.knot[i] == 0

    model.cons1 = Constraint(RangeSet(1, K - 1), rule=cons1_rule)

    return model


# ToDo: pass but try with linesearch
def create_model146():
    # gouldqp3 QLR2-MN-V-V
    model = aml.ConcreteModel()
    model._name = 'gouldqp3'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    K = 350

    def alpha_rule(model, i):
        if i > 1:
            return 1.0 + 1.01 ** i
        else:
            return 2.0

    model.alpha = Param(RangeSet(1, K + 1), initialize=alpha_rule)

    def knot_init_rule(model, i):
        return model.alpha[i]

    def knot_bounds_rule(model, i):
        return (model.alpha[i], model.alpha[i + 1])

    model.knot = Var(RangeSet(1, K), initialize=knot_init_rule, bounds=knot_bounds_rule)

    def space_init_rule(model, i):
        return model.alpha[i + 1] - model.alpha[i]

    def space_bounds_rule(model, i):
        return (0.4 * (model.alpha[i + 2] - model.alpha[i]), 0.6 * (model.alpha[i + 2] - model.alpha[i]))

    model.space = Var(RangeSet(1, K - 1), initialize=space_init_rule, bounds=space_bounds_rule)

    def f_rule(model):
        return sum(.5 * (model.space[i + 1] - model.space[i]) ** 2 for i in range(1, K - 1)) + \
               sum(0.5 * (model.knot[K - i] + model.space[i] - model.alpha[K + 1 - i]) ** 2 for i in range(1, K))

    model.f = aml.Objective(rule=f_rule)

    def cons1_rule(model, i):
        return model.space[i] - model.knot[i + 1] + model.knot[i] == 0

    model.cons1 = Constraint(RangeSet(1, K - 1), rule=cons1_rule)

    return model


def create_model147():

    # gpp OOR2-AY-V-0
    model = aml.ConcreteModel()
    model._name = 'gpp'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    n = 250

    model.x = Var(RangeSet(1, n), initialize=1.0)

    def f_rule(model):
        return sum(aml.exp(model.x[j] - model.x[i]) for i in range(1, n) for j in range(i + 1, n + 1))

    model.f = aml.Objective(rule=f_rule)

    def cons1_rule(model, i):
        return model.x[i] + model.x[i + 1] >= 0.0

    model.cons1 = Constraint(RangeSet(1, n - 1), rule=cons1_rule)

    def cons2_rule(model, i):
        return aml.exp(model.x[i]) + aml.exp(model.x[i + 1]) <= 20.0

    model.cons2 = Constraint(RangeSet(1, n - 1), rule=cons2_rule)

    return model


def create_model148():

    # growth NOR2-AN-3-12
    model = aml.ConcreteModel()
    model._name = 'growth'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    log = aml.log

    n = 3

    model.u1 = Var(initialize=100.0)
    model.u2 = Var()
    model.u3 = Var()

    def obj_rule(model):
        return (model.u1 * (8.0 ** (model.u2 + (log(8.0)) * model.u3)) - 8.0) ** 2 + \
               (model.u1 * (9.0 ** (model.u2 + (log(9.0)) * model.u3)) - \
                8.4305) ** 2 + (model.u1 * (10.0 ** (model.u2 + (log(10.0)) * model.u3)) - 9.5294) ** 2 + (model.u1 * \
                                                                                                           (11.0 ** (
                                                                                                           model.u2 + (
                                                                                                           log(
                                                                                                               11.0)) * model.u3)) - 10.4627) ** 2 + \
               (model.u1 * (12.0 ** (model.u2 + (log(12.0)) * model.u3)) - \
                12.0) ** 2 + (model.u1 * (13.0 ** (model.u2 + (log(13.0)) * model.u3)) - 13.0205) ** 2 + (model.u1 * \
                                                                                                          (14.0 ** (
                                                                                                          model.u2 + (
                                                                                                          log(
                                                                                                              14.0)) * model.u3)) - 14.5949) ** 2 + \
               (model.u1 * (15.0 ** (model.u2 + (log(15.0)) * model.u3)) - \
                16.1078) ** 2 + (model.u1 * (16.0 ** (model.u2 + (log(16.0)) * model.u3)) - 18.0596) ** 2 + (model.u1 * \
                                                                                                             (18.0 ** (
                                                                                                             model.u2 + (
                                                                                                             log(
                                                                                                                 18.0)) * model.u3)) - 20.4569) ** 2 + \
               (model.u1 * (20.0 ** (model.u2 + (log(20.0)) * model.u3)) - \
                24.25) ** 2 + (model.u1 * (25.0 ** (model.u2 + (log(25.0)) * model.u3)) - 32.9863) ** 2

    model.obj = aml.Objective(rule=obj_rule)

    return model


def create_model149():

    # growthls SUR2-AN-3-0
    model = aml.ConcreteModel()
    model._name = 'growthls'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    log = aml.log

    n = 3

    model.u1 = Var(initialize=100.0)
    model.u2 = Var()
    model.u3 = Var()

    def obj_rule(model):
        return (model.u1 * (8.0 ** (model.u2 + (log(8.0)) * model.u3)) - 8.0) * (
        model.u1 * (8.0 ** (model.u2 + (log(8.0)) * model.u3)) - 8.0) + \
               (model.u1 * (9.0 ** (model.u2 + (log(9.0)) * model.u3)) - 8.4305) * (
               model.u1 * (9.0 ** (model.u2 + (log(9.0)) * model.u3)) - \
               8.4305) + (model.u1 * (10.0 ** (model.u2 + (log(10.0)) * model.u3)) - 9.5294) * (model.u1 * \
                                                                                                (10.0 ** (model.u2 + (
                                                                                                log(
                                                                                                    10.0)) * model.u3)) - 9.5294) + (
                                                                                                                                    model.u1 * (
                                                                                                                                    11.0 ** (
                                                                                                                                    model.u2 + (
                                                                                                                                    log(
                                                                                                                                        11.0)) * model.u3)) - \
                                                                                                                                    10.4627) * (
                                                                                                                                    model.u1 * (
                                                                                                                                    11.0 ** (
                                                                                                                                    model.u2 + (
                                                                                                                                    log(
                                                                                                                                        11.0)) * model.u3)) - 10.4627) + (
                                                                                                                                                                         model.u1 * \
                                                                                                                                                                         (
                                                                                                                                                                         12.0 ** (
                                                                                                                                                                         model.u2 + (
                                                                                                                                                                         log(
                                                                                                                                                                             12.0)) * model.u3)) - 12.0) * (
                                                                                                                                                                         model.u1 * (
                                                                                                                                                                         12.0 ** (
                                                                                                                                                                         model.u2 + (
                                                                                                                                                                         log(
                                                                                                                                                                             12.0)) * model.u3)) - 12.0) + \
               (model.u1 * (13.0 ** (model.u2 + (log(13.0)) * model.u3)) - 13.0205) * (
               model.u1 * (13.0 ** (model.u2 + (log(13.0)) * model.u3)) - \
               13.0205) + (model.u1 * (14.0 ** (model.u2 + (log(14.0)) * model.u3)) - 14.5949) * (model.u1 * \
                                                                                                  (14.0 ** (model.u2 + (
                                                                                                  log(
                                                                                                      14.0)) * model.u3)) - 14.5949) + (
                                                                                                                                       model.u1 * (
                                                                                                                                       15.0 ** (
                                                                                                                                       model.u2 + (
                                                                                                                                       log(
                                                                                                                                           15.0)) * model.u3)) - \
                                                                                                                                       16.1078) * (
                                                                                                                                       model.u1 * (
                                                                                                                                       15.0 ** (
                                                                                                                                       model.u2 + (
                                                                                                                                       log(
                                                                                                                                           15.0)) * model.u3)) - 16.1078) + (
                                                                                                                                                                            model.u1 * \
                                                                                                                                                                            (
                                                                                                                                                                            16.0 ** (
                                                                                                                                                                            model.u2 + (
                                                                                                                                                                            log(
                                                                                                                                                                                16.0)) * model.u3)) - 18.0596) * (
                                                                                                                                                                            model.u1 * (
                                                                                                                                                                            16.0 ** (
                                                                                                                                                                            model.u2 + (
                                                                                                                                                                            log(
                                                                                                                                                                                16.0)) * model.u3)) - \
                                                                                                                                                                            18.0596) + (
                                                                                                                                                                                       model.u1 * (
                                                                                                                                                                                       18.0 ** (
                                                                                                                                                                                       model.u2 + (
                                                                                                                                                                                       log(
                                                                                                                                                                                           18.0)) * model.u3)) - 20.4569) * (
                                                                                                                                                                                       model.u1 * \
                                                                                                                                                                                       (
                                                                                                                                                                                       18.0 ** (
                                                                                                                                                                                       model.u2 + (
                                                                                                                                                                                       log(
                                                                                                                                                                                           18.0)) * model.u3)) - 20.4569) + (
                                                                                                                                                                                                                            model.u1 * (
                                                                                                                                                                                                                            20.0 ** (
                                                                                                                                                                                                                            model.u2 + (
                                                                                                                                                                                                                            log(
                                                                                                                                                                                                                                20.0)) * model.u3)) - \
                                                                                                                                                                                                                            24.25) * (
                                                                                                                                                                                                                            model.u1 * (
                                                                                                                                                                                                                            20.0 ** (
                                                                                                                                                                                                                            model.u2 + (
                                                                                                                                                                                                                            log(
                                                                                                                                                                                                                                20.0)) * model.u3)) - 24.25) + (
                                                                                                                                                                                                                                                               model.u1 * \
                                                                                                                                                                                                                                                               (
                                                                                                                                                                                                                                                               25.0 ** (
                                                                                                                                                                                                                                                               model.u2 + (
                                                                                                                                                                                                                                                               log(
                                                                                                                                                                                                                                                                   25.0)) * model.u3)) - 32.9863) * (
                                                                                                                                                                                                                                                               model.u1 * (
                                                                                                                                                                                                                                                               25.0 ** (
                                                                                                                                                                                                                                                               model.u2 + (
                                                                                                                                                                                                                                                               log(
                                                                                                                                                                                                                                                                   25.0)) * model.u3)) - \
                                                                                                                                                                                                                                                               32.9863)

    model.obj = aml.Objective(rule=obj_rule)
    return model


def create_model150():

    # gulf SUR2-MN-3-0
    model = aml.ConcreteModel()
    model._name = 'gulf'
    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    log = aml.log
    N = 3
    M = 99

    def t_param_rule(model, i):
        return i / 100.0

    model.t = Param(RangeSet(1, M), initialize=t_param_rule)

    def y_param_rule(model, i):
        return 25 + (-50 * log(aml.value(model.t[i]))) ** (2.0 / 3.0)

    model.y = Param(RangeSet(1, M), initialize=y_param_rule)

    model.x = Var(RangeSet(1, N))
    model.x[1] = 5.0
    model.x[2] = 2.5
    model.x[3] = 0.15

    def f_rule(model):
        return sum((aml.exp(abs(model.y[i] - model.x[2]) ** model.x[3] / (-model.x[1])) - model.t[i]) ** 2 for i in
                   range(1, M + 1))

    model.f = aml.Objective(rule=f_rule)
    return model


def create_model151():

    # hager1 SLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'hager1'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.N = Param(initialize=5000)
    model.Sx = RangeSet(0, model.N)
    model.Sy = RangeSet(1, model.N)
    model.x = Var(model.Sx, initialize=0.0)
    model.u = Var(model.Sy, initialize=0.0)

    def f(m):
        sum_exprs = []
        sum_coefs = []
        for i in m.Sy:
            sum_exprs.append((m.u[i] ** 2) / (2 * value(m.N)))
        expr = 0.5 * m.x[value(m.N)] ** 2 + sum(sum_exprs)
        return expr

    model.f = aml.Objective(rule=f)

    def cons1(m, i):
        expr = (value(m.N) - 0.5) * m.x[i] + (-value(m.N) - 0.5) * m.x[i - 1] - m.u[i]
        return (0, expr)

    model.cons1 = Constraint(model.Sy, rule=cons1)

    def cons2(m):
        return (1.0, m.x[0])

    model.cons2 = Constraint(rule=cons2)
    return model


def create_model152():

    # hager2 OLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'hager2'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.n = Param(initialize=5000)
    model.h = Param(initialize=1.0 / 5000.0)

    model.Sx = RangeSet(0, model.n)
    model.Su = RangeSet(1, model.n)

    model.x = Var(model.Sx, initialize=0.0)
    model.u = Var(model.Su, initialize=0.0)

    def f(m):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in m.Su:
            sum_expr_1 += (value(m.h)) * (m.x[i - 1] ** 2 + m.x[i - 1] * m.x[i] + m.x[i] ** 2) / 6
            sum_expr_2 += ((value(m.h)) * (m.u[i]) ** 2) / 4
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(m, i):
        exp = (value(m.n) - 0.25) * m.x[i] - (value(m.n) + 0.25) * m.x[i - 1] - m.u[i]
        return (0, exp)

    model.cons1 = Constraint(model.Su, rule=cons1)

    model.x[0] = 1.0
    model.x[0].fixed = True
    return model


def create_model153():

    # hager3 SLR2-AY-V-V
    model = aml.ConcreteModel()
    model._name = 'hager3'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.n = Param(initialize=5000.0)
    model.h = Param(initialize=1.0 / 5000.0)

    model.Sx = RangeSet(0, model.n)
    model.Su = RangeSet(1, model.n)

    model.x = Var(model.Sx, initialize=0.0)
    model.u = Var(model.Su, initialize=0.0)

    def f(m):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in m.Su:
            sum_expr_1 += (value(m.h)) * (
            0.625 * (m.x[i - 1] ** 2 + m.x[i - 1] * m.x[i] + m.x[i] ** 2) + ((m.x[i - 1] + m.x[i]) * m.u[i] \
                                                                             )) / 8
            sum_expr_2 += (value(m.h) * (m.u[i]) ** 2) / 4
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(m, i):
        exp = (value(m.n) - 0.25) * m.x[i] - (value(m.n) + 0.25) * m.x[i - 1] - m.u[i]
        return (0, exp)

    model.cons1 = Constraint(model.Su, rule=cons1)

    model.x[0] = 1.0
    model.x[0].fixed = True
    return model


# ToDo: fail try with line search
def create_model154():

    # hager 4 OLR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'hager4'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    def t_rule(m, i):
        return i * value(m.h)

    def z_rule(m, i):
        return aml.exp(-2 * value(m.t[i]))

    def a_rule(m, i):
        return -0.5 * value(m.z[i])

    def b_rule(m, i):
        return value(m.a[i]) * (value(m.t[i]) + 0.5)

    def c_rule(m, i):
        return value(m.a[i]) * (value(m.t[i]) ** 2 + value(m.t[i]) + 0.5)

    def scda_rule(m):
        return (value(m.a[1]) - value(m.a[0])) / 2

    def scdb_rule(m):
        return (value(m.b[1]) - value(m.b[0])) * value(m.n)

    def scdc_rule(m):
        return (value(m.c[1]) - value(m.c[0])) * value(m.n) * value(m.n) * 0.5

    def xx0_rule(m):
        return (1 + 3 * value(m.e)) / (2 - 2 * value(m.e))

    model.n = Param(initialize=5000)
    model.h = Param(initialize=1.0 / 5000.0)
    model.one = Param(initialize=1)

    model.Sx = RangeSet(0, model.n)
    model.Su = RangeSet(1, model.n)
    model.S = RangeSet(0, model.one)

    model.t = Param(model.Sx, initialize=t_rule)
    model.z = Param(model.Sx, initialize=z_rule)
    model.a = Param(model.S, initialize=a_rule)
    model.b = Param(model.S, initialize=b_rule)
    model.c = Param(model.S, initialize=c_rule)
    model.scda = Param(initialize=scda_rule)
    model.scdb = Param(initialize=scdb_rule)
    model.scdc = Param(initialize=scdc_rule)
    model.e = Param(initialize=aml.exp(1))
    model.xx0 = Param(initialize=xx0_rule)

    def u_bounds(m, i):
        return (None, 1.0)

    model.x = Var(model.Sx, initialize=0.0)
    model.u = Var(model.Su, bounds=u_bounds, initialize=0.0)

    def f(m):
        sum_expr_1 = 0
        sum_expr_2 = 0
        for i in m.Su:
            sum_expr_1 += (value(m.scda) * m.z[i - 1] * m.x[i] ** 2 + value(m.scdb) * m.z[i - 1] * m.x[i] * (
            m.x[i - 1] - m.x[i]) + value(m.scdc) * m.z[i - 1] * (m.x[i - 1] - m.x[i]) ** 2)
            sum_expr_2 += ((value(m.h)) * (m.u[i]) ** 2) * 0.5
        exp = sum_expr_1 + sum_expr_2
        return exp

    model.f = aml.Objective(rule=f)

    def cons1(m, i):
        return (0, (value(m.n) - 1) * m.x[i] - value(m.n) * m.x[i - 1] - aml.exp(value(m.t[i])) * m.u[i])

    model.cons1 = Constraint(model.Su, rule=cons1)

    model.x[0].fix(value(model.xx0))
    return model


def create_model155():

    # hairy OUR2-AY-2-0
    model = aml.ConcreteModel()
    model._name = 'hairy'


    hlength = 30
    cslope = 100

    model.x1 = aml.Var(initialize=-5)
    model.x2 = aml.Var(initialize=-7)

    def f_rule(model):
        return aml.sin(7 * model.x1) ** 2 * aml.cos(7 * model.x2) ** 2 * hlength + \
               cslope * aml.sqrt(0.01 + (model.x1 - model.x2) ** 2) + cslope * aml.sqrt(0.01 + model.x1 ** 2)

    model.f = aml.Objective(rule=f_rule)
    return model


# ToDo: fail try with line search
def create_model156():

    # haldmads LOR2-AN-6-42
    model = aml.ConcreteModel()
    model._name = 'haldmads'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    def y_rule(model, i):
        return -1.0 + 0.1 * (i - 1)

    model.y = Param(RangeSet(1, 21), initialize=y_rule)

    def ey_rule(model, i):
        return aml.exp(model.y[i])

    model.ey = Param(RangeSet(1, 21), initialize=ey_rule)

    def x_init_rule(model, i):
        if i == 1:
            return 0.5
        else:
            return 0.0

    model.x = Var(RangeSet(1, 5), initialize=x_init_rule)

    model.u = Var(initialize=0.0)

    model.f = aml.Objective(expr=model.u)

    def cons1_rule(model, i):
        return (model.x[1] + model.y[i] * model.x[2]) / (1.0 + model.x[3] * model.y[i] + \
        model.x[4] * model.y[i] ** 2 + model.x[5] * model.y[i] ** 3) - model.u <= model.ey[i]

    model.cons1 = Constraint(RangeSet(1, 21), rule=cons1_rule)

    def cons2_rule(model, i):
        return -(model.x[1] + model.y[i] * model.x[2]) / (1.0 + model.x[3] * model.y[i] + \
        model.x[4] * model.y[i] ** 2 + model.x[5] * model.y[i] ** 3) - model.u <= -model.ey[i]

    model.cons2 = Constraint(RangeSet(1, 21), rule=cons2_rule)

    return model


def create_model157():

    # harkerp2 QBR2-AN-V-V
    model = aml.ConcreteModel()
    model._name = 'harkerp2'

    N=100

    def x_init_rule(model,i):
        return i
    model.x = aml.Var(aml.RangeSet(1,N),bounds=(0.0,None),initialize=x_init_rule)

    def f_rule(model):
        return sum (-1*model.x[i]**2*0.5 for i in range(1,N+1)) +\
        sum(-model.x[i]for i in range(1,N+1)) +\
        (sum(model.x[i] for i in range(1,N+1)))**2+\
        sum (2*(sum(model.x[i] for i in range(j,N+1)))**2 for j in range(2,N+1))
    model.f = aml.Objective(rule=f_rule)
    return model


def create_model158():
    import numpy as np
    # hart6 OBR2-AN-6-0
    model = aml.ConcreteModel()
    model._name = 'hart6'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.c = dict()
    model.c[1] = 1.0
    model.c[2] = 1.2
    model.c[3] = 3.0
    model.c[4] = 3.2

    model.a = np.zeros((5, 7))

    model.a[1, 1] = 10.0
    model.a[1, 2] = 0.05
    model.a[1, 3] = 17.0
    model.a[1, 4] = 3.05
    model.a[1, 5] = 1.7
    model.a[1, 6] = 8.0
    model.a[2, 1] = 0.05
    model.a[2, 2] = 10.0
    model.a[2, 3] = 17.0
    model.a[2, 4] = 0.1
    model.a[2, 5] = 8.0
    model.a[2, 6] = 14.0
    model.a[3, 1] = 3.0
    model.a[3, 2] = 3.5
    model.a[3, 3] = 1.7
    model.a[3, 4] = 10.0
    model.a[3, 5] = 17.0
    model.a[3, 6] = 8.0
    model.a[4, 1] = 17.0
    model.a[4, 2] = 8.0
    model.a[4, 3] = 0.05
    model.a[4, 4] = 10.0
    model.a[4, 5] = 0.1
    model.a[4, 6] = 14.0

    model.p = np.zeros((5, 7))
    model.p[1, 1] = 0.1312
    model.p[1, 2] = 0.1696
    model.p[1, 3] = 0.5569
    model.p[1, 4] = 0.0124
    model.p[1, 5] = 0.8283
    model.p[1, 6] = 0.5886

    model.p[2, 1] = 0.2329
    model.p[2, 2] = 0.4135
    model.p[2, 3] = 0.8307
    model.p[2, 4] = 0.3736
    model.p[2, 5] = 0.1004
    model.p[2, 6] = 0.9991

    model.p[3, 1] = 0.2348
    model.p[3, 2] = 0.1451
    model.p[3, 3] = 0.3522
    model.p[3, 4] = 0.2883
    model.p[3, 5] = 0.3047
    model.p[3, 6] = 0.6650

    model.p[4, 1] = 0.4047
    model.p[4, 2] = 0.8828
    model.p[4, 3] = 0.8732
    model.p[4, 4] = 0.5743
    model.p[4, 5] = 0.1091
    model.p[4, 6] = 0.0381

    model.x = Var(RangeSet(1, 6), bounds=(0.0, 1.0), initialize=0.2)

    def obj_rule(model):
        return - sum(model.c[i] * aml.exp(-sum(model.a[i, j] * (model.x[j] - model.p[i, j]) ** 2 for j in range(1, 7))) for i in range(1, 5))

    model.obj = aml.Objective(rule=obj_rule)
    model.pprint()
    return model


def create_model159():

    # hatflda SBR2-AN-4-0
    model = aml.ConcreteModel()
    model._name = 'hatflda'


    N = 4
    model.x = aml.Var(aml.RangeSet(1, N), bounds=(0.0000001, None), initialize=0.1)

    def f_rule(model):
        return (model.x[1] - 1) ** 2 + sum((model.x[i - 1] - (model.x[i]) ** 0.5) ** 2 for i in range(2, N + 1))

    model.f = aml.Objective(rule=f_rule)

    return model


# failed in first iteration singular kkt
def create_model160():

    # hs055
    model = aml.ConcreteModel()
    model._name = 'hs055'

    model.x = aml.Var(range(1, 7), within=aml.NonNegativeReals, initialize=0.0)
    model.x[1].setub(1)
    model.x[4].setub(1)

    model.x[1] = 1.0
    model.x[2] = 2.0
    model.x[6] = 2.0

    model.obj = aml.Objective(expr=model.x[1] + 2.0 * model.x[2] + 4.0 * model.x[5] + aml.exp(model.x[1] * model.x[4]))

    model.constr1 = aml.Constraint(expr=model.x[1] + 2 * model.x[2] + 5 * model.x[5] == 6)
    model.constr2 = aml.Constraint(expr=model.x[1] + model.x[2] + model.x[3] == 3)
    model.constr3 = aml.Constraint(expr=model.x[4] + model.x[5] + model.x[6] == 2)
    model.constr4 = aml.Constraint(expr=model.x[1] + model.x[4] == 1)
    model.constr5 = aml.Constraint(expr=model.x[2] + model.x[5] == 2)
    model.constr6 = aml.Constraint(expr=model.x[3] + model.x[6] == 2)

    return model


def create_model161():

    # hs056
    model = aml.ConcreteModel()
    model._name = 'hs056'

    model.N = aml.RangeSet(1, 7)
    model.x = aml.Var(model.N, bounds=(0, None))

    model.x[1] = 1
    model.x[2] = 1
    model.x[3] = 1
    model.x[4] = aml.asin(aml.sqrt(1 / 4.2))
    model.x[5] = aml.asin(aml.sqrt(1 / 4.2))
    model.x[6] = aml.asin(aml.sqrt(1 / 4.2))
    model.x[7] = aml.asin(aml.sqrt(5 / 7.2))

    model.obj = aml.Objective(expr=-model.x[1] * model.x[2] * model.x[3])

    model.constr1 = aml.Constraint(expr=model.x[1] - 4.2 * aml.sin(model.x[4]) ** 2 == 0)
    model.constr2 = aml.Constraint(expr=model.x[2] - 4.2 * aml.sin(model.x[5]) ** 2 == 0)
    model.constr3 = aml.Constraint(expr=model.x[3] - 4.2 * aml.sin(model.x[6]) ** 2 == 0)
    model.constr4 = aml.Constraint(expr=model.x[1] + 2 * model.x[2] + 2 * model.x[3] - 7.2 * aml.sin(model.x[7]) ** 2 == 0)

    return model


# ToDo: fails try with line search
def create_model162():

    # hs059
    model = aml.ConcreteModel()
    model._name = 'hs059'

    model.N = aml.RangeSet(1, 2)
    model.x = aml.Var(model.N, within=aml.NonNegativeReals)
    model.x[1].setub(75.0)
    model.x[2].setub(65.0)

    model.x[1] = 90.0
    model.x[2] = 10.0

    model.obj = aml.Objective(expr=-75.196 + 3.8112 * model.x[1] + 0.0020567 * model.x[1] ** 3 - 1.0345e-5 * model.x[1] ** 4 \
                               + 6.8306 * model.x[2] - 0.030234 * model.x[1] * model.x[2] + 1.28134e-3 * model.x[2] *
                               model.x[1] ** 2 + 2.266e-7 * model.x[1] ** 4 * model.x[2] - 0.25645 * model.x[2] ** 2 + 0.0034604 *
                               model.x[2] ** 3 - 1.3514e-5 * model.x[2] ** 4 \
                               + 28.106 / (model.x[2] + 1.0) + 5.2375e-6 * model.x[1] ** 2 * model.x[2] ** 2 + 6.3e-8 *
                               model.x[1] ** 3 * model.x[2] ** 2 \
                               - 7e-10 * model.x[1] ** 3 * model.x[2] ** 3 - 3.405e-4 * model.x[1] * model.x[2] ** 2 + 1.6638e-6 *
                               model.x[1] * model.x[2] ** 3 \
                               + 2.8673 * aml.exp(0.0005 * model.x[1] * model.x[2]) - 3.5256e-5 * model.x[1] ** 3 * model.x[2] \
                               # the last term appears in CUTE but not in H&S
                               - 0.12694 * model.x[1] ** 2)

    model.constr1 = aml.Constraint(expr=model.x[1] * model.x[2] >= 700)
    model.constr2 = aml.Constraint(expr=model.x[2] - (model.x[1] ** 2) / 125.0 >= 0)
    model.constr3 = aml.Constraint(expr=(model.x[2] - 50) ** 2 - 5 * (model.x[1] - 55) >= 0)
    return model


def create_model163():

    # hs060
    model = aml.ConcreteModel()
    model._name = 'hs060'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N, initialize=2.0, bounds=(-10, 10))

    model.obj = aml.Objective(expr=(model.x[1] - 1) ** 2 + (model.x[1] - model.x[2]) ** 2 + (model.x[2] - model.x[3]) ** 4)

    model.constr1 = aml.Constraint(expr=model.x[1] * (1 + model.x[2] ** 2) + model.x[3] ** 4 == 4 + 3 * aml.sqrt(2))

    return model


# ToDo: fails try with line search
def create_model164():

    # hs061
    model = aml.ConcreteModel()
    model._name = 'hs061'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N, initialize=0.0)

    model.obj = aml.Objective(
        expr=4 * model.x[1] ** 2 + 2 * model.x[2] ** 2 + 2 * model.x[3] ** 2 - 33 * model.x[1] + 16 * model.x[2] \
             - 24 * model.x[3])

    model.constr1 = aml.Constraint(expr=3 * model.x[1] - 2 * model.x[2] ** 2 == 7)
    model.constr2 = aml.Constraint(expr=4 * model.x[1] - model.x[3] ** 2 == 11)

    return model


def create_model165():
    # hs062
    model = aml.ConcreteModel()
    model._name = 'hs062'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N, bounds=(0, 1))
    model.x[1] = 0.7
    model.x[2] = 0.2
    model.x[3] = 0.1

    model.obj = aml.Objective(
        expr=-32.174 * (255 * aml.log((model.x[1] + model.x[2] + model.x[3] + 0.03) / (0.09 * model.x[1] + model.x[2] \
                        + model.x[3] + 0.03)) \
                        + 280 * aml.log((model.x[2] + model.x[3] + 0.03) / (0.07 * model.x[2] + model.x[3] + 0.03)) \
                        + 290 * aml.log((model.x[3] + 0.03) / (0.13 * model.x[3] + 0.03))))

    model.constr1 = aml.Constraint(expr=model.x[1] + model.x[2] + model.x[3] == 1)

    return model


def create_model166():

    # hs063
    model = aml.ConcreteModel()
    model._name = 'hs063'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N, within=aml.NonNegativeReals, initialize=2)

    model.obj = aml.Objective(expr=1000 - model.x[1] ** 2 - 2 * model.x[2] ** 2 - \
                                   model.x[3] ** 2 - model.x[1] * model.x[2] - model.x[1] * model.x[3])

    model.constr1 = aml.Constraint(expr=8 * model.x[1] + 14 * model.x[2] + 7 * model.x[3] == 56)
    model.constr2 = aml.Constraint(expr=model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 == 25)

    return model


# ToDo: fails try with line search
def create_model167():

    # hs064
    model = aml.ConcreteModel()
    model._name = 'hs064'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N, initialize=1.0, bounds=(1.0e-5, None))

    model.obj = aml.Objective(expr=5 * model.x[1] + 50000.0 / model.x[1] + 20 * model.x[2] + 72000.0 / model.x[2] \
                               + 10 * model.x[3] + 144000.0 / model.x[3])

    model.constr1 = aml.Constraint(expr=4.0 / model.x[1] + 32.0 / model.x[2] + 120.0 / model.x[3] <= 1)
    return model


def create_model168():

    # hs065
    model = aml.ConcreteModel()
    model._name = 'hs065'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N)
    model.x[1] = -5
    model.x[2] = 5
    model.x[3] = 0

    model.obj = aml.Objective(expr=(model.x[1] - model.x[2]) ** 2 + (model.x[1] + model.x[2] - 10.0) ** 2 / 9.0 \
                               + (model.x[3] - 5) ** 2)

    model.constr1 = aml.Constraint(expr=model.x[1] ** 2 + model.x[2] ** 2 + model.x[3] ** 2 <= 48)
    model.constr2 = aml.Constraint(expr=-4.5 <= model.x[1] <= 4.5)
    model.constr3 = aml.Constraint(expr=-4.5 <= model.x[2] <= 4.5)
    model.constr4 = aml.Constraint(expr=-5 <= model.x[3] <= 5)

    return model


def create_model169():

    # hs066
    model = aml.ConcreteModel()
    model._name = 'hs066'

    model.N = aml.RangeSet(1, 3)
    model.x = aml.Var(model.N)
    model.x[1] = 0
    model.x[2] = 1.05
    model.x[3] = 2.9

    model.obj = aml.Objective(expr=0.2 * model.x[3] - 0.8 * model.x[1])

    model.constr1 = aml.Constraint(expr=model.x[2] - aml.exp(model.x[1]) >= 0)
    model.constr2 = aml.Constraint(expr=model.x[3] - aml.exp(model.x[2]) >= 0)
    model.constr3 = aml.Constraint(expr=0 <= model.x[1] <= 100)
    model.constr4 = aml.Constraint(expr=0 <= model.x[2] <= 100)
    model.constr5 = aml.Constraint(expr=0 <= model.x[3] <= 10)

    return model


def create_model170():

    # hs067
    model = aml.ConcreteModel()
    model._name = 'hs067'

    model.N = aml.RangeSet(1, 8)
    model.M = aml.RangeSet(1, 14)
    model.x1 = aml.Var(bounds=(1.0e-5, 2.0e+3))
    model.x2 = aml.Var(bounds=(1.0e-5, 1.6e+4))
    model.x3 = aml.Var(bounds=(1.0e-5, 1.2e+2))
    model.y = aml.Var(model.N, initialize=0.0)
    model.a = aml.Param(model.M, mutable=True)
    model.x1 = 1745.0
    model.x2 = 12000.0
    model.x3 = 110.0
    model.a[1] = 0.0
    model.a[2] = 0.0
    model.a[3] = 85.0
    model.a[4] = 90.0
    model.a[5] = 3.0
    model.a[6] = 0.01
    model.a[7] = 145.0
    model.a[8] = 5000.0
    model.a[9] = 2000.0
    model.a[10] = 93.0
    model.a[11] = 95.0
    model.a[12] = 12.0
    model.a[13] = 4.0
    model.a[14] = 162.0

    model.obj = aml.Objective(
        expr=-(0.063 * model.y[2] * model.y[5] - 5.04 * model.x1 - 3.36 * model.y[3] - 0.035 * model.x2 - 10 * model.x3))

    model.constr1 = aml.Constraint(aml.RangeSet(1, 7), rule=lambda model, i: model.y[i + 1] >= model.a[i])
    model.constr2 = aml.Constraint(aml.RangeSet(8, 14), rule=lambda model, i: model.a[i] >= model.y[i - 6])
    model.constr3 = aml.Constraint(expr=model.y[3] == 1.22 * model.y[2] - model.x1)
    model.constr4 = aml.Constraint(expr=model.y[6] == (model.x2 + model.y[3]) / model.x1)
    model.constr5 = aml.Constraint(expr=model.y[2] == 0.01 * model.x1 * (112 + 13.167 * model.y[6] - 0.6667 * model.y[6] ** 2))
    model.constr6 = aml.Constraint(
        expr=model.y[5] == 86.35 + 1.098 * model.y[6] - 0.038 * model.y[6] ** 2 + 0.325 * (model.y[4] - 89))
    model.constr7 = aml.Constraint(expr=model.y[8] == 3.0 * model.y[5] - 133.0)
    model.constr8 = aml.Constraint(expr=model.y[7] == 35.82 - 0.222 * model.y[8])
    model.constr9 = aml.Constraint(expr=model.y[4] == 98000.0 * model.x3 / (model.y[2] * model.y[7] + 1000.0 * model.x3))

    value = aml.value
    model.y[2] = 1.6 * value(model.x1)
    while (1):
        y2old = model.y[2].value
        model.y[3] = 1.22 * model.y[2].value - model.x1.value
        model.y[6] = (model.x2.value + model.y[3].value) / model.x1.value
        model.y[2] = 0.01 * model.x1.value * \
                     (112.0 + \
                      13.167 * model.y[6].value - \
                      0.6667 * model.y[6].value ** 2)
        if abs(y2old - model.y[2].value) < 0.001:
            break

    model.y[4] = 93.0
    while (1):
        y4old = model.y[4].value
        model.y[5] = 86.35 + \
                     1.098 * model.y[6].value - \
                     0.038 * model.y[6].value ** 2 + \
                     0.325 * (model.y[4].value - 89)
        model.y[8] = 3 * model.y[5].value - 133
        model.y[7] = 35.82 - 0.222 * model.y[8].value
        model.y[4] = 98000.0 * model.x3.value / \
                     (model.y[2].value * model.y[7].value + 1000 * model.x3.value)
        if abs(y4old - model.y[4].value) < 0.001:
            break

    return model


def create_model171():

    # hs071
    model = aml.ConcreteModel()
    model._name = 'hs071'

    model.N = aml.RangeSet(1, 4)
    model.x = aml.Var(model.N, bounds=(1, 5))
    model.x[1] = 1
    model.x[2] = 5
    model.x[3] = 5
    model.x[4] = 1

    model.obj = aml.Objective(expr=model.x[1] * model.x[4] * (model.x[1] + model.x[2] + model.x[3]) + model.x[3])

    def cons1_rule(model):
        expr = 1.0
        for i in range(1, 5):
            expr *= model.x[i]
        return expr >= 25

    model.constr1 = aml.Constraint(rule=cons1_rule)

    # model.constr1 = Constraint(expr=prod {i in 1..4} x[i] >= 25

    model.constr2 = aml.Constraint(expr=sum(model.x[i] ** 2 for i in model.N) == 40)
    return model


# ToDo: fails try with line search
def create_model172():
    import numpy as np

    # hs072
    model = aml.ConcreteModel()
    model._name = 'hs072'

    model.N = aml.RangeSet(1, 2)
    model.M = aml.RangeSet(1, 4)
    model.x = aml.Var(model.M, bounds=(0.001, None), initialize=1.0)
    model.a = np.zeros((3, 5))

    model.a[1, 1] = 4.0
    model.a[1, 2] = 2.25
    model.a[1, 3] = 1.0
    model.a[1, 4] = 0.25

    model.a[2, 1] = 0.16
    model.a[2, 2] = 0.36
    model.a[2, 3] = 0.64
    model.a[2, 4] = 0.64

    model.b = dict()
    model.b[1] = 0.0401
    model.b[2] = 0.010085

    def obj_rule(model):
        return 1 + sum(model.x[j] for j in model.M)

    model.obj = aml.Objective(rule=obj_rule)

    def cons_rule(model, i):
        return sum((model.a[i, j] / model.x[j]) for j in model.M) <= model.b[i]

    model.constr = aml.Constraint(model.N, rule=cons_rule)

    def ub_rule(model, j):
        return model.x[j] <= (5 - j) * 1.0e5

    model.ub = aml.Constraint(model.M, rule=ub_rule)
    return model


def create_model173():

    # hs073
    model = aml.ConcreteModel()
    model._name = 'hs073'

    model.N = aml.RangeSet(1, 4)
    model.x = aml.Var(model.N, initialize=1.0, within=aml.NonNegativeReals)

    model.obj = aml.Objective(expr=24.55 * model.x[1] + 26.75 * model.x[2] + 39 * model.x[3] + 40.50 * model.x[4])

    model.constr1 = aml.Constraint(expr=2.3 * model.x[1] + 5.6 * model.x[2] + 11.1 * model.x[3] + 1.3 * model.x[4] >= 5)
    model.constr2 = aml.Constraint(
        expr=-(21.0 + 1.645 * aml.sqrt(0.28 * model.x[1] ** 2 + 0.19 * model.x[2] ** 2 + 20.5 * model.x[3] ** 2 + \
                                   0.62 * model.x[4] ** 2)) \
             + (12.0 * model.x[1] + 11.9 * model.x[2] + 41.8 * model.x[3] + 52.1 * model.x[4]) \
             >= 0.0)
    model.constr3 = aml.Constraint(expr=sum(model.x[j] for j in model.N) == 1)

    return model


def create_model174():

    # hs074
    model = aml.ConcreteModel()
    model._name = 'hs074'

    model.a = 0.55
    model.N = aml.RangeSet(1, 4)
    model.x = aml.Var(model.N, initialize=0.0)
    model.x[1].setub(1200)
    model.x[1].setlb(0)
    model.x[2].setub(1200)
    model.x[2].setlb(0)
    model.x[3].setub(model.a)
    model.x[3].setlb(-model.a)
    model.x[4].setub(model.a)
    model.x[4].setlb(-model.a)

    model.obj = aml.Objective(
        expr=3.0 * model.x[1] + 1.0e-6 * model.x[1] ** 3 + 2 * model.x[2] + 2.0e-6 * model.x[2] ** 3 / 3)

    model.constr1 = aml.Constraint(expr=-model.a <= model.x[4] - model.x[3] <= model.a)
    model.constr2 = aml.Constraint(
        expr=model.x[1] == 1000 * aml.sin(-model.x[3] - 0.25) + 1000 * aml.sin(-model.x[4] - 0.25) + 894.8)
    model.constr3 = aml.Constraint(
        expr=model.x[2] == 1000 * aml.sin(model.x[3] - 0.25) + 1000 * aml.sin(model.x[3] - model.x[4] - 0.25) + 894.8)
    model.constr4 = aml.Constraint(
        expr=0.0 == 1000 * aml.sin(model.x[4] - 0.25) + 1000 * aml.sin(model.x[4] - model.x[3] - 0.25) + 1294.8)

    return model


def create_model175():

    # hs075
    model = aml.ConcreteModel()
    model._name = 'hs075'

    model.a = 0.48
    model.N = aml.RangeSet(1, 4)
    model.x = aml.Var(model.N, initialize=0.0)
    model.x[1].setub(1200)
    model.x[1].setlb(0)
    model.x[2].setub(1200)
    model.x[2].setlb(0)
    model.x[3].setub(model.a)
    model.x[3].setlb(-model.a)
    model.x[4].setub(model.a)
    model.x[4].setlb(-model.a)

    model.obj = aml.Objective(
        expr=3.0 * model.x[1] + 1.0e-6 * model.x[1] ** 3 + 2 * model.x[2] + 2.0e-6 * model.x[2] ** 3 / 3)

    model.constr1 = aml.Constraint(
        expr=-model.a <= model.x[4] - model.x[3] <= model.a)
    model.constr2 = aml.Constraint(
        expr=model.x[1] == 1000 * aml.sin(-model.x[3] - 0.25) + 1000 * aml.sin(-model.x[4] - 0.25) + 894.8)
    model.constr3 = aml.Constraint(
        expr=model.x[2] == 1000 * aml.sin(model.x[3] - 0.25) + 1000 * aml.sin(model.x[3] - model.x[4] - 0.25) + 894.8)
    model.constr4 = aml.Constraint(
        expr=0.0 == 1000 * aml.sin(model.x[4] - 0.25) + 1000 * aml.sin(model.x[4] - model.x[3] - 0.25) + 1294.8)

    return model


def create_model176():

    # hs076
    model = aml.ConcreteModel()
    model._name = 'hs076'

    model.N = aml.RangeSet(1, 4)
    model.x = aml.Var(model.N, initialize=0.5, within=aml.NonNegativeReals)

    model.obj = aml.Objective(expr=model.x[1] ** 2 + 0.5 * model.x[2] ** 2 + model.x[3] ** 2 + 0.5 * model.x[4] ** 2 \
                               - model.x[1] * model.x[3] + model.x[3] * model.x[4] \
                               - model.x[1] - 3 * model.x[2] + model.x[3] - model.x[4])

    model.constr1 = aml.Constraint(expr=model.x[1] + 2.0 * model.x[2] + model.x[3] + model.x[4] <= 5.0)
    model.constr2 = aml.Constraint(expr=3.0 * model.x[1] + model.x[2] + 2.0 * model.x[3] - model.x[4] <= 4.0)
    model.constr3 = aml.Constraint(expr=model.x[2] + 4 * model.x[3] >= 1.5)

    return model


def create_model177():

    # hs077
    model = aml.ConcreteModel()
    model._name = 'hs077'

    model.N = aml.RangeSet(1, 5)
    model.x = aml.Var(model.N, initialize=2.0)

    model.obj = aml.Objective(expr=(model.x[1] - 1) ** 2 + (model.x[1] - model.x[2]) ** 2 \
                               + (model.x[3] - 1) ** 2 + (model.x[4] - 1) ** 4 + (model.x[5] - 1) ** 6)

    model.constr1 = aml.Constraint(expr=model.x[1] ** 2 * model.x[4] + aml.sin(model.x[4] - model.x[5]) == 2 * aml.sqrt(2))
    model.constr2 = aml.Constraint(expr=model.x[2] + model.x[3] ** 4 * model.x[4] ** 2 == 8 + aml.sqrt(2))

    return model


def create_model178():

    # hs078
    model = aml.ConcreteModel()
    model._name = 'hs078'

    model.N = aml.RangeSet(1, 5)
    model.x = aml.Var(model.N)
    model.x[1] = -2
    model.x[2] = 1.5
    model.x[3] = 2
    model.x[4] = -1
    model.x[5] = -1

    def obj_rule(model):
        expr = 1.0
        for i in model.N:
            expr *= model.x[i]
        return expr

    model.obj = aml.Objective(rule=obj_rule)

    model.constr1 = aml.Constraint(expr=sum(model.x[j] ** 2 for j in model.N) == 10)
    model.constr2 = aml.Constraint(expr=model.x[2] * model.x[3] - 5 * model.x[4] * model.x[5] == 0)
    model.constr3 = aml.Constraint(expr=model.x[1] ** 3 + model.x[2] ** 3 == -1)

    return model


def create_model179():

    # hs079
    model = aml.ConcreteModel()
    model._name = 'hs079'

    model.N = aml.RangeSet(1, 5)
    model.x = aml.Var(model.N, initialize=2.0)

    model.obj = aml.Objective(expr=(model.x[1] - 1) ** 2 + (model.x[1] - model.x[2]) ** 2 + (model.x[2] - model.x[3]) ** 2 \
                               + (model.x[3] - model.x[4]) ** 4 + (model.x[4] - model.x[5]) ** 4)

    model.constr1 = aml.Constraint(expr=model.x[1] + model.x[2] ** 2 + model.x[3] ** 3 == 2 + 3 * aml.sqrt(2))
    model.constr2 = aml.Constraint(expr=model.x[2] - model.x[3] ** 2 + model.x[4] == -2 + 2 * aml.sqrt(2))
    model.constr3 = aml.Constraint(expr=model.x[1] * model.x[5] == 2)

    return model


def create_model180():

    # hs080
    model = aml.ConcreteModel()
    model._name = 'hs080'

    model.N = aml.RangeSet(1, 5)
    model.x = aml.Var(model.N)
    model.x[1] = -2.0
    model.x[2] = 2.0
    model.x[3] = 2.0
    model.x[4] = -1.0
    model.x[5] = -1.0
    model.a = 2.3
    model.b = 3.2
    model.x[1].setub(model.a)
    model.x[1].setlb(-model.a)
    model.x[2].setub(model.a)
    model.x[2].setlb(-model.a)
    model.x[3].setub(model.b)
    model.x[3].setlb(-model.b)
    model.x[4].setub(model.b)
    model.x[4].setlb(-model.b)
    model.x[5].setub(model.b)
    model.x[5].setlb(-model.b)

    def obj_rule(model):
        expr = 1.0
        for j in model.N:
            expr *= model.x[j]
        return aml.exp(expr)

    model.obj = aml.Objective(rule=obj_rule)

    model.constr1 = aml.Constraint(expr=sum(model.x[j] ** 2 for j in model.N) == 10)
    model.constr2 = aml.Constraint(expr=model.x[2] * model.x[3] - 5 * model.x[4] * model.x[5] == 0)
    model.constr3 = aml.Constraint(expr=model.x[1] ** 3 + model.x[2] ** 3 == -1)

    return model


def create_model181():

    # hs081
    model = aml.ConcreteModel()
    model._name = 'hs081'

    model.N = aml.RangeSet(1, 5)
    model.x = aml.Var(model.N)
    model.x[1] = -2.0
    model.x[2] = 2.0
    model.x[3] = 2.0
    model.x[4] = -1.0
    model.x[5] = -1.0
    model.a = 2.3
    model.b = 3.2
    model.x[1].setub(model.a)
    model.x[1].setlb(-model.a)
    model.x[2].setub(model.a)
    model.x[2].setlb(-model.a)
    model.x[3].setub(model.b)
    model.x[3].setlb(-model.b)
    model.x[4].setub(model.b)
    model.x[4].setlb(-model.b)
    model.x[5].setub(model.b)
    model.x[5].setlb(-model.b)

    def obj_rule(model):
        expr = 1.0
        for j in model.N:
            expr *= model.x[j]
        return aml.exp(expr) - 0.5 * (model.x[1] ** 3 + model.x[2] ** 3 + 1) ** 2

    model.obj = aml.Objective(rule=obj_rule)

    model.constr1 = aml.Constraint(expr=sum(model.x[j] ** 2 for j in model.N) == 10)
    model.constr2 = aml.Constraint(expr=model.x[2] * model.x[3] - 5 * model.x[4] * model.x[5] == 0)
    model.constr3 = aml.Constraint(expr=model.x[1] ** 3 + model.x[2] ** 3 == -1)

    return model


def create_model182():

    # hs083
    model = aml.ConcreteModel()
    model._name = 'hs083'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.N = RangeSet(1, 5)
    model.M = RangeSet(1, 12)
    model.a = dict()
    model.l = dict()
    model.u = dict()

    model.a[1] = 85.334407
    model.a[2] = 0.0056858
    model.a[3] = 0.0006262
    model.a[4] = 0.0022053
    model.a[5] = 80.51249
    model.a[6] = 0.0071317
    model.a[7] = 0.0029955
    model.a[8] = 0.0021813
    model.a[9] = 9.300961
    model.a[10] = 0.0047026
    model.a[11] = 0.0012547
    model.a[12] = 0.0019085

    model.l[1] = 78.0
    model.l[2] = 33.0
    model.l[3] = 27.0
    model.l[4] = 27.0
    model.l[5] = 27.0

    model.u[1] = 102.0
    model.u[2] = 45.0
    model.u[3] = 45.0
    model.u[4] = 45.0
    model.u[5] = 45.0


    x_init = {}
    x_init[1] = 78.0
    x_init[2] = 33.0
    x_init[3] = 27.0
    x_init[4] = 27.0
    x_init[5] = 27.0

    def x_init_rule(model, i):
        return x_init[i]

    def x_bounds_rule(model, i):
        return (value(model.l[i]), value(model.u[i]))

    model.x = Var(model.N, bounds=x_bounds_rule, initialize=x_init_rule)

    def obj_rule(model):
        return 5.3578547 * model.x[3] ** 2 + 0.8356891 * model.x[1] * model.x[5] + 37.293239 * model.x[1] - 40792.141
    model.obj = aml.Objective(rule=obj_rule)

    def cons1_rule(model):
        return 0 <= model.a[1] + model.a[2] * model.x[2] * model.x[5] + model.a[3] * model.x[1] * model.x[4] \
                    - model.a[4] * model.x[3] * model.x[5] <= 92
    model.constr1 = Constraint(rule=cons1_rule)

    def cons2_rule(model):
        return 0 <= model.a[5] + model.a[6] * model.x[2] * model.x[5] + model.a[7] * model.x[1] * model.x[2] \
                    + model.a[8] * model.x[3] ** 2 - 90 <= 20
    model.constr2 = Constraint(rule=cons2_rule)

    def cons3_rule(model):
        return 0 <= model.a[9] + model.a[10] * model.x[3] * model.x[5] + model.a[11] * model.x[1] * model.x[3] \
                    + model.a[12] * model.x[3] * model.x[4] - 20 <= 5
    model.constr3 = Constraint(rule=cons3_rule)

    return model


def create_model183():

    # hs084
    model = aml.ConcreteModel()
    model._name = 'hs084'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint

    model.N = RangeSet(1, 5)
    model.M = RangeSet(1, 21)
    model.l = dict()
    model.u = dict()
    model.a = dict()

    model.a[1] = -24345.0
    model.a[2] = -8720288.849
    model.a[3] = 150512.5253
    model.a[4] = -156.6950325
    model.a[5] = 476470.3222
    model.a[6] = 729482.8271
    model.a[7] = -145421.402
    model.a[8] = 2931.1506
    model.a[9] = -40.427932
    model.a[10] = 5106.192
    model.a[11] = 15711.36
    model.a[12] = -155011.1084
    model.a[13] = 4360.53352
    model.a[14] = 12.9492344
    model.a[15] = 10236.884
    model.a[16] = 13176.786
    model.a[17] = -326669.5104
    model.a[18] = 7390.68412
    model.a[19] = -27.8986976
    model.a[20] = 16643.076
    model.a[21] = 30988.146

    model.l[1] = 0.0
    model.l[2] = 1.2
    model.l[3] = 20.0
    model.l[4] = 9.0
    model.l[5] = 6.5

    model.u[1] = 1000.0
    model.u[2] = 2.4
    model.u[3] = 60.0
    model.u[4] = 9.3
    model.u[5] = 7.0

    x_init = {}
    x_init[1] = 2.52
    x_init[2] = 2.0
    x_init[3] = 37.5
    x_init[4] = 9.25
    x_init[5] = 6.8

    def x_bounds_rule(model, i):
        return (value(model.l[i]), value(model.u[i]))

    def x_init_rule(model, i):
        return x_init[i]

    model.x = Var(model.N, bounds=x_bounds_rule, initialize=x_init_rule)

    def obj_rule(model):
        return -model.a[1] - model.a[2] * model.x[1] - model.a[3] * model.x[1] * model.x[2] \
               - model.a[4] * model.x[1] * model.x[3] - model.a[5] * model.x[1] * model.x[4] \
               - model.a[6] * model.x[1] * model.x[5]
    model.obj = aml.Objective(rule=obj_rule)

    def cons1_rule(model):
        return 0 <= model.a[7] * model.x[1] + model.a[8] * model.x[1] * model.x[2] \
                    + model.a[9] * model.x[1] * model.x[3] + model.a[10] * model.x[1] * model.x[4] \
                    + model.a[11] * model.x[1] * model.x[5] <= 294000.0
    model.constr1 = Constraint(rule=cons1_rule)

    def cons2_rule(model):
        return 0 <= model.a[12] * model.x[1] + model.a[13] * model.x[1] * model.x[2] \
                    + model.a[14] * model.x[1] * model.x[3] + model.a[15] * model.x[1] * model.x[4] \
                    + model.a[16] * model.x[1] * model.x[5] <= 294000.0
    model.constr2 = Constraint(rule=cons2_rule)

    def cons3_rule(model):
        return 0 <= model.a[17] * model.x[1] + model.a[18] * model.x[1] * model.x[2] \
                    + model.a[19] * model.x[1] * model.x[3] + model.a[20] * model.x[1] * model.x[4] \
                    + model.a[21] * model.x[1] * model.x[5] <= 277200.0
    model.constr3 = Constraint(rule=cons3_rule)

    return model


def create_model184():

    # hs085
    model = aml.ConcreteModel()
    model._name = 'hs085'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    a = {}
    a[2] = 17.505
    a[3] = 11.275
    a[4] = 214.228
    a[5] = 7.458
    a[6] = .961
    a[7] = 1.612
    a[8] = .146
    a[9] = 107.99
    a[10] = 922.693
    a[11] = 926.832
    a[12] = 18.766
    a[13] = 1072.163
    a[14] = 8961.448
    a[15] = .063
    a[16] = 71084.33
    a[17] = 2802713.0

    b = {}
    b[2] = 1053.6667
    b[3] = 35.03
    b[4] = 665.585
    b[5] = 584.463
    b[6] = 265.916
    b[7] = 7.046
    b[8] = .222
    b[9] = 273.366
    b[10] = 1286.105
    b[11] = 1444.046
    b[12] = 537.141
    b[13] = 3247.039
    b[14] = 26844.086
    b[15] = .386
    b[16] = 140000.0
    b[17] = 12146108

    c10 = 12.3 / 752.3

    model.x = Var(range(1, 6))
    model.x[1] = 900.0
    model.x[2] = 80.0
    model.x[3] = 115.0
    model.x[4] = 267.0
    model.x[5] = 27.0

    model.y1 = Expression(initialize=model.x[2] + model.x[3] + 41.6)
    model.c1 = Expression(initialize=.024 * model.x[4] - 4.62)
    model.y2 = Expression(initialize=12.5 / model.c1 + 12)
    model.c2 = Expression(initialize=.0003535 * model.x[1] ** 2 + .5311 * model.x[1] + .08705 * model.y2 * model.x[1])
    model.c3 = Expression(initialize=.052 * model.x[1] + 78 + .002377 * model.y2 * model.x[1])
    model.y3 = Expression(initialize=model.c2 / model.c3)
    model.y4 = Expression(initialize=19. * model.y3)
    model.c4 = Expression(
        initialize=.04782 * (model.x[1] - model.y3) + .1956 * (model.x[1] - model.y3) ** 2. / model.x[2] + .6376 * \
        model.y4 + 1.594 * model.y3)
    model.c5 = Expression(initialize=100. * model.x[2])
    model.c6 = Expression(initialize=model.x[1] - model.y3 - model.y4)
    model.c7 = Expression(initialize=.95 - model.c4 / model.c5)
    model.y5 = Expression(initialize=model.c6 * model.c7)
    model.y6 = Expression(initialize=model.x[1] - model.y5 - model.y4 - model.y3)
    model.c8 = Expression(initialize=(model.y5 + model.y4) * .995)
    model.y7 = Expression(initialize=model.c8 / model.y1)
    model.y8 = Expression(initialize=model.c8 / 3798.)
    model.c9 = Expression(initialize=model.y7 - .0663 * model.y7 / model.y8 - .3153)
    model.y9 = Expression(initialize=96.82 / model.c9 + .321 * model.y1)
    model.y10 = Expression(initialize=1.29 * model.y5 + 1.258 * model.y4 + 2.29 * model.y3 + 1.71 * model.y6)
    model.y11 = Expression(initialize=1.71 * model.x[1] - .452 * model.y4 + .58 * model.y3)
    model.c11 = Expression(initialize=1.75 * model.y2 * .995 * model.x[1])
    model.c12 = Expression(initialize=.995 * model.y10 + 1998.)
    model.y12 = Expression(initialize=c10 * model.x[1] + model.c11 / model.c12)
    model.y13 = Expression(initialize=model.c12 - 1.75 * model.y2)
    model.y14 = Expression(initialize=3623. + 64.4 * model.x[2] + 58.4 * model.x[3] + 146312. / (model.y9 + model.x[5]))
    model.c13 = Expression(
        initialize=.995 * model.y10 + 60.8 * model.x[2] + 48. * model.x[4] - .1121 * model.y14 - 5095.)
    model.y15 = Expression(initialize=model.y13 / model.c13)
    model.y16 = Expression(initialize=148000. - 331000. * model.y15 + 40. * model.y13 - 61. * model.y15 * model.y13)
    model.c14 = Expression(initialize=2324. * model.y10 - 28740000. * model.y2)
    model.y17 = Expression(initialize=14130000. - 1328. * model.y10 - 531. * model.y11 + model.c14 / model.c12)
    model.c15 = Expression(initialize=model.y13 / model.y15 - model.y13 / .52)
    model.c16 = Expression(initialize=1.104 - .72 * model.y15)
    model.c17 = Expression(initialize=model.y9 + model.x[5])

    model.obj = aml.Objective(
        expr=-5.843e-7 * model.y17 + 1.17e-4 * model.y14 + 2.358e-5 * model.y13 + 1.502e-6 * model.y16 \
        + .0321 * model.y12 + .004324 * model.y5 + 1e-4 * model.c15 / model.c16 + 37.48 * model.y2 / model.c12 + .1365)

    model.con1 = Constraint(expr=1.5 * model.x[2] - model.x[3] >= 0)
    model.con2 = Constraint(expr=model.y1 - 213.1 >= 0)
    model.con3 = Constraint(expr=405.23 - model.y1 >= 0)
    model.con4 = Constraint(expr=model.x[1] >= 704.4148)
    model.con5 = Constraint(expr=model.x[1] <= 906.3855)
    model.con6 = Constraint(expr=model.x[2] >= 68.6)
    model.con7 = Constraint(expr=model.x[2] <= 288.88)
    model.con8 = Constraint(expr=model.x[3] >= 0)
    model.con9 = Constraint(expr=model.x[3] <= 134.75)
    model.con10 = Constraint(expr=model.x[4] >= 193)
    model.con11 = Constraint(expr=model.x[4] <= 287.0966)
    model.con12 = Constraint(expr=model.x[5] >= 25)
    model.con13 = Constraint(expr=model.x[5] <= 84.1988)
    model.con14 = Constraint(expr=model.y2 - a[2] >= 0)
    model.con15 = Constraint(expr=model.y3 - a[3] >= 0)
    model.con16 = Constraint(expr=model.y4 - a[4] >= 0)
    model.con17 = Constraint(expr=model.y5 - a[5] >= 0)
    model.con18 = Constraint(expr=model.y6 - a[6] >= 0)
    model.con19 = Constraint(expr=model.y7 - a[7] >= 0)
    model.con20 = Constraint(expr=model.y8 - a[8] >= 0)
    model.con21 = Constraint(expr=model.y9 - a[9] >= 0)
    model.con22 = Constraint(expr=model.y10 - a[10] >= 0)
    model.con23 = Constraint(expr=model.y11 - a[11] >= 0)
    model.con24 = Constraint(expr=model.y12 - a[12] >= 0)
    model.con25 = Constraint(expr=model.y13 - a[13] >= 0)
    model.con26 = Constraint(expr=model.y14 - a[14] >= 0)
    model.con27 = Constraint(expr=model.y15 - a[15] >= 0)
    model.con28 = Constraint(expr=model.y16 - a[16] >= 0)
    model.con29 = Constraint(expr=model.y17 - a[17] >= 0)
    model.con30 = Constraint(expr=b[2] - model.y2 >= 0)
    model.con31 = Constraint(expr=b[3] - model.y3 >= 0)
    model.con32 = Constraint(expr=b[4] - model.y4 >= 0)
    model.con33 = Constraint(expr=b[5] - model.y5 >= 0)
    model.con34 = Constraint(expr=b[6] - model.y6 >= 0)
    model.con35 = Constraint(expr=b[7] - model.y7 >= 0)
    model.con36 = Constraint(expr=b[8] - model.y8 >= 0)
    model.con37 = Constraint(expr=b[9] - model.y9 >= 0)
    model.con38 = Constraint(expr=b[10] - model.y10 >= 0)
    model.con39 = Constraint(expr=b[11] - model.y11 >= 0)
    model.con40 = Constraint(expr=b[12] - model.y12 >= 0)
    model.con41 = Constraint(expr=b[13] - model.y13 >= 0)
    model.con42 = Constraint(expr=b[14] - model.y14 >= 0)
    model.con43 = Constraint(expr=b[15] - model.y15 >= 0)
    model.con44 = Constraint(expr=b[16] - model.y16 >= 0)
    model.con45 = Constraint(expr=b[17] - model.y17 >= 0)
    model.con46 = Constraint(expr=model.y4 - .28 / .72 * model.y5 >= 0)
    model.con47 = Constraint(expr=21 - 3496. * model.y2 / model.c12 >= 0)
    model.con48 = Constraint(expr=62212. / model.c17 - 110.6 - model.y1 >= 0)

    return model


def create_model185():

    # hs087
    model = aml.ConcreteModel()
    model._name = 'hs087'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.a = 131.078
    model.b = 1.48477
    model.c = 0.90798
    model.d = aml.cos(1.47588)
    model.e = aml.sin(1.47588)
    model.lim1 = 300
    model.lim2 = 100
    model.lim3 = 200
    model.rate1 = 30
    model.rate2 = 31
    model.rate3 = 28
    model.rate4 = 29
    model.rate5 = 30

    model.N = RangeSet(1, 6)

    model.z1 = Var()
    model.z2 = Var()
    model.x1 = Var()
    model.x2 = Var()
    model.x3 = Var()
    model.x4 = Var()
    model.x5 = Var()
    model.x6 = Var()

    model.x1.setlb(0.0)
    model.x1.setub(400.0)
    model.x2.setlb(0.0)
    model.x2.setub(1000.0)
    model.x3.setlb(340.0)
    model.x3.setub(420.0)
    model.x4.setlb(340.0)
    model.x4.setub(420.0)
    model.x5.setlb(-1000.0)
    model.x5.setub(1000.0)
    model.x6.setlb(0.0)
    model.x6.setub(0.5236)

    model.x1 = 390.0
    model.x2 = 1000.0
    model.x3 = 419.5
    model.x4 = 340.5
    model.x5 = 198.175
    model.x6 = 0.5

    model.obj = aml.Objective(expr=model.z1 + model.z2)

    def f1(model, x):
        if x == 0:
            return 0.0
        elif x == 300:
            return 300 * 30.0
        elif x == 400:
            return 30.0 * 300.0 + 31.0 * 100.0

    def f2(model, x):
        if x == 0:
            return 0.0
        elif x == 100:
            return 28.0 * 100.0
        elif x == 200:
            return 28.0 * 100.0 + 29 * 100.0
        elif x == 1000:
            return 28.0 * 100.0 + 29 * 100.0 + 30.0 * 800

    model.piecew1 = aml.Piecewise(model.z1, model.x1, pw_constr_type='LB', pw_pts=[0.0, model.lim1, 400.0], f_rule=f1)
    model.piecew2 = aml.Piecewise(model.z2, model.x2, pw_constr_type='LB', pw_pts=[0.0, model.lim2, model.lim3, 1000.0],
                                  f_rule= f2)

    model.e1 = Constraint(expr=model.x1 == 300 - model.x3 * model.x4 * aml.cos(model.b - model.x6) / model.a \
                                           + model.c * model.x3 ** 2 * model.d / model.a)
    model.e2 = Constraint(expr=model.x2 == -model.x3 * model.x4 * aml.cos(model.b + model.x6) / model.a \
                                           + model.c * model.x4 ** 2 * model.d / model.a)
    model.e3 = Constraint(expr=model.x5 == -model.x3 * model.x4 * aml.sin(model.b + model.x6) / model.a \
                                           + model.c * model.x4 ** 2 * model.e / model.a)
    model.e4 = Constraint(expr=200 - model.x3 * model.x4 * aml.sin(model.b - model.x6) / model.a \
                               + model.c * model.x3 ** 2 * model.e / model.a == 0)

    return model


# ToDo: fail, try with line search
def create_model186():

    # hs088
    model = aml.ConcreteModel()
    model._name = 'hs088'
    model.mu = dict()

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.mu[1] = 8.6033358901938017e-01
    model.mu[2] = 3.4256184594817283e+00
    model.mu[3] = 6.4372981791719468e+00
    model.mu[4] = 9.5293344053619631e+00
    model.mu[5] = 1.2645287223856643e+01
    model.mu[6] = 1.5771284874815882e+01
    model.mu[7] = 1.8902409956860023e+01
    model.mu[8] = 2.2036496727938566e+01
    model.mu[9] = 2.5172446326646664e+01
    model.mu[10] = 2.8309642854452012e+01
    model.mu[11] = 3.1447714637546234e+01
    model.mu[12] = 3.4586424215288922e+01
    model.mu[13] = 3.7725612827776501e+01
    model.mu[14] = 4.0865170330488070e+01
    model.mu[15] = 4.4005017920830845e+01
    model.mu[16] = 4.7145097736761031e+01
    model.mu[17] = 5.0285366337773652e+01
    model.mu[18] = 5.3425790477394663e+01
    model.mu[19] = 5.6566344279821521e+01
    model.mu[20] = 5.9707007305335459e+01
    model.mu[21] = 6.2847763194454451e+01
    model.mu[22] = 6.5988598698490392e+01
    model.mu[23] = 6.9129502973895256e+01
    model.mu[24] = 7.2270467060308960e+01
    model.mu[25] = 7.5411483488848148e+01
    model.mu[26] = 7.8552545984242926e+01
    model.mu[27] = 8.1693649235601683e+01
    model.mu[28] = 8.4834788718042290e+01
    model.mu[29] = 8.7975960552493220e+01
    model.mu[30] = 9.1117161394464745e+01

    n = 2
    model.N = RangeSet(1, 30)
    model.M = RangeSet(1, n)

    def A_rule(model, j):
        return 2.0 * aml.sin(model.mu[j]) / (model.mu[j] + aml.sin(model.mu[j]) * aml.cos(model.mu[j]))

    model.A = Param(model.N, initialize=A_rule)

    def x_init_rule(model, i):
        return 0.5 * (-1) ** (i + 1)

    model.x = Var(model.M, initialize=x_init_rule)

    def rho(model, j):
        return -(aml.exp(-model.mu[j] ** 2 * sum(model.x[i] ** 2 for i in model.M)) + sum(
        2.0 * (-1) ** (ii - 1) * aml.exp(-model.mu[j] ** 2 * sum(model.x[i] ** 2 for i in range(ii, n + 1))) for ii in
        range(2, n + 1)) + (-1.0) ** n) / model.mu[j] ** 2

    def obj_rule(model):
        return sum(model.x[i] ** 2 for i in model.M)

    model.obj = aml.Objective(rule=obj_rule)

    def cons1_rule(model):
        return sum(model.mu[i] ** 2 * model.mu[j] ** 2 * model.A[i] * model.A[j] * rho(model, i) * rho(model, j) * (aml.sin(model.mu[i] + model.mu[j]) / (model.mu[i] + model.mu[j]) + aml.sin(model.mu[i] - model.mu[j]) / (
        model.mu[i] - model.mu[j])) for i in model.N for j in range(i+1, 30+1)) \
        + sum(model.mu[j] ** 4 * model.A[j] ** 2 * rho(model, j) ** 2 * ((aml.sin(2.0 * model.mu[j]) / (2.0 * model.mu[j]) + 1.0) / 2.0) \
        for j in model.N) -sum(model.mu[j] ** 2 * model.A[j] * rho(model, j) * (2.0 * aml.sin(model.mu[j]) / (model.mu[j] ** 3) - 2.0 * aml.cos(model.mu[j]) / (model.mu[j] ** 2)) \
            for j in model.N) <= -2.0 / 15.0 + 0.0001

    model.constr1 = Constraint(rule=cons1_rule)

    return model


#ToDo: fails try with line search
def create_model187():

    # hs089
    model = aml.ConcreteModel()
    model._name = 'hs089'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.mu = dict()
    model.mu[1] = 8.6033358901938017e-01
    model.mu[2] = 3.4256184594817283e+00
    model.mu[3] = 6.4372981791719468e+00
    model.mu[4] = 9.5293344053619631e+00
    model.mu[5] = 1.2645287223856643e+01
    model.mu[6] = 1.5771284874815882e+01
    model.mu[7] = 1.8902409956860023e+01
    model.mu[8] = 2.2036496727938566e+01
    model.mu[9] = 2.5172446326646664e+01
    model.mu[10] = 2.8309642854452012e+01
    model.mu[11] = 3.1447714637546234e+01
    model.mu[12] = 3.4586424215288922e+01
    model.mu[13] = 3.7725612827776501e+01
    model.mu[14] = 4.0865170330488070e+01
    model.mu[15] = 4.4005017920830845e+01
    model.mu[16] = 4.7145097736761031e+01
    model.mu[17] = 5.0285366337773652e+01
    model.mu[18] = 5.3425790477394663e+01
    model.mu[19] = 5.6566344279821521e+01
    model.mu[20] = 5.9707007305335459e+01
    model.mu[21] = 6.2847763194454451e+01
    model.mu[22] = 6.5988598698490392e+01
    model.mu[23] = 6.9129502973895256e+01
    model.mu[24] = 7.2270467060308960e+01
    model.mu[25] = 7.5411483488848148e+01
    model.mu[26] = 7.8552545984242926e+01
    model.mu[27] = 8.1693649235601683e+01
    model.mu[28] = 8.4834788718042290e+01
    model.mu[29] = 8.7975960552493220e+01
    model.mu[30] = 9.1117161394464745e+01

    n = 3
    model.N = RangeSet(1, 30)
    model.M = RangeSet(1, n)

    def A_rule(model, j):
        return 2.0 * aml.sin(model.mu[j]) / (model.mu[j] + aml.sin(model.mu[j]) * aml.cos(model.mu[j]))

    model.A = Param(model.N, initialize=A_rule)

    def x_init_rule(model, i):
        return 0.5 * (-1) ** (i + 1)

    model.x = Var(model.M, initialize=x_init_rule)

    def rho(model, j):
        return -(aml.exp(-model.mu[j] ** 2 * sum(model.x[i] ** 2 for i in model.M)) \
        + sum(2.0 * (-1) ** (ii - 1) * aml.exp(-model.mu[j] ** 2 * sum(model.x[i] ** 2 for i in range(ii, n + 1))) for ii in
        range(2, n + 1)) + (-1) ** n) / model.mu[j] ** 2


    def obj_rule(model):
        return sum(model.x[i] ** 2 for i in model.M)

    model.obj = aml.Objective(rule=obj_rule)

    def cons1_rule(model):
        return sum(model.mu[i] ** 2 * model.mu[j] ** 2 * model.A[i] * model.A[j] * rho(model, i) * rho(model, j) \
                   * (aml.sin(model.mu[i] + model.mu[j]) / (model.mu[i] + model.mu[j]) + aml.sin(model.mu[i] - model.mu[j]) / (
        model.mu[i] - model.mu[j])) for i in model.N for j in range(i + 1, 30 + 1)) \
        + sum(model.mu[j] ** 4 * model.A[j] ** 2 * rho(model, j) ** 2 * (
        aml.sin(2.0 * model.mu[j]) / (2.0 * model.mu[j]) + 1.0) / 2.0 for j in model.N) \
        - sum(model.mu[j] ** 2 * model.A[j] * rho(model, j) * (2.0 * aml.sin(model.mu[j]) / model.mu[j] ** 3 \
        - 2.0 * aml.cos(model.mu[j]) / model.mu[j] ** 2) for j in model.N) + 2.0 / 15.0 <= 0.0001

    model.constr1 = aml.Constraint(rule=cons1_rule)

    return model


def create_model188():

    # hs093
    model = aml.ConcreteModel()
    model._name = 'hs093'

    model.N = aml.RangeSet(1, 6)
    model.x = aml.Var(model.N, bounds=(0, None))
    model.x[1] = 5.54
    model.x[2] = 4.4
    model.x[3] = 12.02
    model.x[4] = 11.82
    model.x[5] = 0.702
    model.x[6] = 0.852

    model.obj = aml.Objective(expr=0.0204 * model.x[1] * model.x[4] * (model.x[1] + model.x[2] + model.x[3]) + \
                               0.0187 * model.x[2] * model.x[3] * (model.x[1] + 1.57 * model.x[2] + model.x[4]) + \
                               0.0607 * model.x[1] * model.x[4] * model.x[5] ** 2 * (
                               model.x[1] + model.x[2] + model.x[3]) + \
                               0.0437 * model.x[2] * model.x[3] * model.x[6] ** 2 * (
                               model.x[1] + 1.57 * model.x[2] + model.x[4]))

    def prod(model):
        expr = 1.0
        for j in model.N:
            expr *= model.x[j]
        return expr

    model.constr1 = aml.Constraint(expr=0.001 * prod(model) >= 2.07)
    model.constr2 = aml.Constraint(expr=0.00062 * model.x[1] * model.x[4] * model.x[5] ** 2 \
                                    * (model.x[1] + model.x[2] + model.x[3]) \
                                    + 0.00058 * model.x[2] * model.x[3] * model.x[6] ** 2 * (
                                    model.x[1] + 1.57 * model.x[2] + model.x[4]) <= 1)

    return model


def create_model189():

    #hs095
    model = aml.ConcreteModel()
    model._name = 'hs095'

    model.N = aml.RangeSet(1, 6)
    model.u = aml.Param(model.N, mutable=True)
    model.u[1] = 0.31
    model.u[2] = 0.046
    model.u[3] = 0.068
    model.u[4] = 0.042
    model.u[5] = 0.028
    model.u[6] = 0.0134

    def x_bounds_rule(model, j):
        return (0, aml.value(model.u[j]))

    model.x = aml.Var(model.N, bounds=x_bounds_rule, initialize=0.0)

    model.obj = aml.Objective(expr=4.3 * model.x[1] + 31.8 * model.x[2] + 63.3 * model.x[3] \
                               + 15.8 * model.x[4] + 68.5 * model.x[5] + 4.7 * model.x[6])

    model.constr1 = aml.Constraint(expr=17.1 * model.x[1] + 38.2 * model.x[2] + 204.2 * model.x[3] \
                                    + 212.3 * model.x[4] + 623.4 * model.x[5] + 1495.5 * model.x[6] \
                                    - 169.0 * model.x[1] * model.x[3] - 3580.0 * model.x[3] * model.x[5] \
                                    - 3810.0 * model.x[4] * model.x[5] - 18500.0 * model.x[4] * model.x[6]
                                    - 24300.0 * model.x[5] * model.x[6] >= 4.97)
    model.constr2 = aml.Constraint(expr=17.9 * model.x[1] + 36.8 * model.x[2] + 113.9 * model.x[3] \
                                    + 169.7 * model.x[4] + 337.8 * model.x[5] + 1385.2 * model.x[6] \
                                    - 139.0 * model.x[1] * model.x[3] - 2450.0 * model.x[4] * model.x[5] - 16600.0 *
                                                                                                           model.x[4] *
                                                                                                           model.x[6] \
                                    - 17200.0 * model.x[5] * model.x[6] >= -1.88)
    model.constr3 = aml.Constraint(expr=-273.0 * model.x[2] - 70.0 * model.x[4] - 819.0 * model.x[5] \
                                    + 26000.0 * model.x[4] * model.x[5] >= -29.08)
    model.constr4 = aml.Constraint(expr=159.9 * model.x[1] - 311.0 * model.x[2] + 587.0 * model.x[4] \
                                    + 391.0 * model.x[5] + 2198.0 * model.x[6] - 14000.0 * model.x[1] * model.x[6] >= -78.02)

    return model


def create_model190():

    # hs096
    model = aml.ConcreteModel()
    model._name = 'hs096'

    model.N = aml.RangeSet(1, 6)
    model.u = aml.Param(model.N, mutable=True)
    model.u[1] = 0.31
    model.u[2] = 0.046
    model.u[3] = 0.068
    model.u[4] = 0.042
    model.u[5] = 0.028
    model.u[6] = 0.0134

    def x_bounds_rule(model, j):
        return (0, aml.value(model.u[j]))

    model.x = aml.Var(model.N, bounds=x_bounds_rule, initialize=0.0)

    model.obj = aml.Objective(expr=4.3 * model.x[1] + 31.8 * model.x[2] + 63.3 * model.x[3] \
                               + 15.8 * model.x[4] + 68.5 * model.x[5] + 4.7 * model.x[6])

    model.constr1 = aml.Constraint(expr=17.1 * model.x[1] + 38.2 * model.x[2] + 204.2 * model.x[3] \
                                    + 212.3 * model.x[4] + 623.4 * model.x[5] + 1495.5 * model.x[6] \
                                    - 169.0 * model.x[1] * model.x[3] - 3580.0 * model.x[3] * model.x[5] \
                                    - 3810.0 * model.x[4] * model.x[5] - 18500.0 * model.x[4] * model.x[6]
                                    - 24300.0 * model.x[5] * model.x[6] >= 4.97)
    model.constr2 = aml.Constraint(expr=17.9 * model.x[1] + 36.8 * model.x[2] + 113.9 * model.x[3] \
                                    + 169.7 * model.x[4] + 337.8 * model.x[5] + 1385.2 * model.x[6] \
                                    - 139.0 * model.x[1] * model.x[3] - 2450.0 * model.x[4] * model.x[5] - 16600.0 *
                                                                                                           model.x[4] *
                                                                                                           model.x[6] \
                                    - 17200.0 * model.x[5] * model.x[6] >= -1.88)
    model.constr3 = aml.Constraint(expr=-273.0 * model.x[2] - 70.0 * model.x[4] - 819.0 * model.x[5] \
                                    + 26000.0 * model.x[4] * model.x[5] >= -69.08)
    model.constr4 = aml.Constraint(expr=159.9 * model.x[1] - 311.0 * model.x[2] + 587.0 * model.x[4] \
                                    + 391.0 * model.x[5] + 2198.0 * model.x[6] - 14000.0 * model.x[1] * model.x[6] >= -118.02)

    return model


def create_model191():

    # hs097
    model = aml.ConcreteModel()
    model._name = 'hs097'


    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.N = RangeSet(1, 6)
    model.u = Param(model.N, mutable=True)
    model.u[1] = 0.31
    model.u[2] = 0.046
    model.u[3] = 0.068
    model.u[4] = 0.042
    model.u[5] = 0.028
    model.u[6] = 0.0134

    def x_bounds_rule(model, j):
        return (0, value(model.u[j]))

    model.x = Var(model.N, bounds=x_bounds_rule, initialize=0.0)

    model.obj = aml.Objective(expr=4.3 * model.x[1] + 31.8 * model.x[2] + 63.3 * model.x[3] \
                               + 15.8 * model.x[4] + 68.5 * model.x[5] + 4.7 * model.x[6])

    model.constr1 = Constraint(expr=17.1 * model.x[1] + 38.2 * model.x[2] + 204.2 * model.x[3] \
                                    + 212.3 * model.x[4] + 623.4 * model.x[5] + 1495.5 * model.x[6] \
                                    - 169.0 * model.x[1] * model.x[3] - 3580.0 * model.x[3] * model.x[5] \
                                    - 3810.0 * model.x[4] * model.x[5] - 18500.0 * model.x[4] * model.x[6]
                                    - 24300.0 * model.x[5] * model.x[6] >= 32.97)
    model.constr2 = Constraint(expr=17.9 * model.x[1] + 36.8 * model.x[2] + 113.9 * model.x[3] \
                                    + 169.7 * model.x[4] + 337.8 * model.x[5] + 1385.2 * model.x[6] \
                                    - 139.0 * model.x[1] * model.x[3] - 2450.0 * model.x[4] * model.x[5] - 16600.0 *
                                                                                                           model.x[4] *
                                                                                                           model.x[6] \
                                    - 17200.0 * model.x[5] * model.x[6] >= 25.12)
    model.constr3 = Constraint(expr=-273.0 * model.x[2] - 70.0 * model.x[4] - 819.0 * model.x[5] \
                                    + 26000.0 * model.x[4] * model.x[5] >= -29.08)
    model.constr4 = Constraint(expr=159.9 * model.x[1] - 311.0 * model.x[2] + 587.0 * model.x[4] \
                                    + 391.0 * model.x[5] + 2198.0 * model.x[6] - 14000.0 * model.x[1] * model.x[6] >= -78.02)

    return model


def create_model192():

    # hs098
    model = aml.ConcreteModel()
    model._name = 'hs098'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.N = RangeSet(1, 6)
    model.u = Param(model.N, mutable=True)
    model.u[1] = 0.31
    model.u[2] = 0.046
    model.u[3] = 0.068
    model.u[4] = 0.042
    model.u[5] = 0.028
    model.u[6] = 0.0134

    def x_bounds_rule(model, j):
        return (0, value(model.u[j]))

    model.x = Var(model.N, bounds=x_bounds_rule, initialize=0.0)

    model.obj = aml.Objective(expr=4.3 * model.x[1] + 31.8 * model.x[2] + 63.3 * model.x[3] \
                               + 15.8 * model.x[4] + 68.5 * model.x[5] + 4.7 * model.x[6])

    model.constr1 = Constraint(expr=17.1 * model.x[1] + 38.2 * model.x[2] + 204.2 * model.x[3] \
                                    + 212.3 * model.x[4] + 623.4 * model.x[5] + 1495.5 * model.x[6] \
                                    - 169.0 * model.x[1] * model.x[3] - 3580.0 * model.x[3] * model.x[5] \
                                    - 3810.0 * model.x[4] * model.x[5] - 18500.0 * model.x[4] * model.x[6]
                                    - 24300.0 * model.x[5] * model.x[6] >= 32.97)
    model.constr2 = Constraint(expr=17.9 * model.x[1] + 36.8 * model.x[2] + 113.9 * model.x[3] \
                                    + 169.7 * model.x[4] + 337.8 * model.x[5] + 1385.2 * model.x[6] \
                                    - 139.0 * model.x[1] * model.x[3] - 2450.0 * model.x[4] * model.x[5] - 16600.0 *
                                    model.x[4] * model.x[6] - 17200.0 * model.x[5] * model.x[6] >= 25.12)
    model.constr3 = Constraint(expr=-273.0 * model.x[2] - 70.0 * model.x[4] - 819.0 * model.x[5] \
                                    + 26000.0 * model.x[4] * model.x[5] >= -124.08)
    model.constr4 = Constraint(expr=159.9 * model.x[1] - 311.0 * model.x[2] + 587.0 * model.x[4] \
                                    + 391.0 * model.x[5] + 2198.0 * model.x[6] - 14000.0 * model.x[1] * model.x[6] >= -173.02)

    return model


def create_model193():

    # hs099
    model = aml.ConcreteModel()
    model._name = 'hs099'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.N = RangeSet(1, 8)
    model.M = RangeSet(1, 7)
    model.L = RangeSet(2, 8)
    model.a = dict()
    model.t = dict()
    model.b = 32.0
    model.x = Var(model.M, bounds=(0, 1.58), initialize=0.5)
    model.q = Var(model.N)
    model.s = Var(model.N)

    model.a[1] = 0
    model.a[2] = 50
    model.a[3] = 50
    model.a[4] = 75
    model.a[5] = 75
    model.a[6] = 75
    model.a[7] = 100
    model.a[8] = 100

    model.t[1] = 0
    model.t[2] = 25
    model.t[3] = 50
    model.t[4] = 100
    model.t[5] = 150
    model.t[6] = 200
    model.t[7] = 290
    model.t[8] = 380

    def obj_rule(model):
        return -(sum(model.a[j + 1] * (model.t[j + 1] - model.t[j]) * aml.cos(model.x[j]) for j in model.M) ** 2)

    model.obj = aml.Objective(rule=obj_rule)

    def cons1_rule(model):
        return model.q[8] == 1.0e+5

    model.constr1 = Constraint(rule=cons1_rule)

    def cons2_rule(model):
        return model.s[8] == 1.0e+3

    model.constr2 = Constraint(rule=cons2_rule)

    def cons3_rule(model):
        return model.q[1] == 0.0

    model.constr3 = Constraint(rule=cons3_rule)

    def cons4_rule(model):
        return model.s[1] == 0.0

    model.constr4 = Constraint(rule=cons4_rule)

    def cons5_rule(model, i):
        return model.q[i] == 0.5 * (model.t[i] - model.t[i - 1]) ** 2 * (model.a[i] * aml.sin(model.x[i - 1]) - model.b) \
                             + (model.t[i] - model.t[i - 1]) * model.s[i - 1] + model.q[i - 1]

    model.constr5 = Constraint(model.L, rule=cons5_rule)

    def cons6_rule(model, i):
        return model.s[i] == (model.t[i] - model.t[i - 1]) * (model.a[i] * aml.sin(model.x[i - 1]) - model.b) + model.s[i - 1]

    model.constr6 = Constraint(model.L, rule=cons6_rule)

    def cons7_rule(model):
        return model.r[1] == 0

    # model.constr7 = Constraint(rule=cons7_rule)

    def cons8_rule(model, i):
        return model.r[i] == model.a[i] * (model.t[i] - model.t[i - 1]) * aml.cos(model.x[i - 1]) + model.r[i - 1]

    # model.constr8 = Constraint(range(2,9),rule=cons8_rule)

    return model


# ToDo: fails, try with line search
def create_model194():

    # hs100
    model = aml.ConcreteModel()
    model._name = 'hs100'

    model.N = aml.RangeSet(1, 7)
    model.x = aml.Var(model.N)
    model.x[1] = 1.0
    model.x[2] = 2.0
    model.x[3] = 0.0
    model.x[4] = 4.0
    model.x[5] = 0.0
    model.x[6] = 1.0
    model.x[7] = 1.0

    model.obj = aml.Objective(
        expr=(model.x[1] - 10) ** 2 + 5.0 * (model.x[2] - 12) ** 2 + model.x[3] ** 4 + 3.0 * (model.x[4] - 11) ** 2 + 10.0 * model.x[5] ** 6 + 7.0 * model.x[6] ** 2 +
        model.x[7] ** 4 - 4.0 * model.x[6] * model.x[7] - 10.0 * model.x[6] - 8.0 * model.x[7])

    model.constr1 = aml.Constraint(
        expr=2 * model.x[1] ** 2 + 3 * model.x[2] ** 4 + model.x[3] + 4 * model.x[4] ** 2 + 5 * model.x[5] <= 127.0)
    model.constr2 = aml.Constraint(
        expr=7 * model.x[1] + 3 * model.x[2] + 10 * model.x[3] ** 2 + model.x[4] - model.x[5] <= 282.0)
    model.constr3 = aml.Constraint(expr=23 * model.x[1] + model.x[2] ** 2 + 6 * model.x[6] ** 2 - 8 * model.x[7] <= 196.0)
    model.constr4 = aml.Constraint(expr=-4 * model.x[1] ** 2 - model.x[2] ** 2 + 3 * model.x[1] * model.x[2] \
                                        - 2 * model.x[3] ** 2 - 5 * model.x[6] + 11 * model.x[7] >= 0)

    return model


def create_model195():

    # hs100lnp OOR2-AN-7-2
    model = aml.ConcreteModel()
    model._name = 'hs100lnp'

    model.N = aml.RangeSet(1, 7)
    model.x = aml.Var(model.N)
    model.x[1] = 1.0
    model.x[2] = 2.0
    model.x[3] = 0.0
    model.x[4] = 4.0
    model.x[5] = 0.0
    model.x[6] = 1.0
    model.x[7] = 1.0

    model.obj = aml.Objective(
        expr=(model.x[1] - 10) ** 2 + 5.0 * (model.x[2] - 12) ** 2 + model.x[3] ** 4 \
        + 3.0 * (model.x[4] - 11) ** 2 + 10.0 * model.x[5] ** 6 + 7.0 * model.x[6] ** 2 + model.x[7] ** 4 \
        - 4.0 * model.x[6] * model.x[7] - 10.0 * model.x[6] - 8.0 * model.x[7])

    model.constr1 = aml.Constraint(
        expr=2 * model.x[1] ** 2 + 3 * model.x[2] ** 4 + model.x[3] + 4 * model.x[4] ** 2 + 5 * model.x[5] == 127.0)
    model.constr4 = aml.Constraint(expr=-4 * model.x[1] ** 2 - model.x[2] ** 2 + 3 * model.x[1] * model.x[2] \
                                    - 2 * model.x[3] ** 2 - 5 * model.x[6] + 11 * model.x[7] == 0)

    return model


# ToDo: fails, try with line search
def create_model196():

    # hs101
    model = aml.ConcreteModel()
    model._name = 'hs101'

    model.N = aml.RangeSet(1, 7)
    model.l = aml.Param(model.N, mutable=True)
    model.a = aml.Param(initialize=-0.25)
    model.l[1] = 0.1
    model.l[2] = 0.1
    model.l[3] = 0.1
    model.l[4] = 0.1
    model.l[5] = 0.1
    model.l[6] = 0.1
    model.l[7] = 0.001

    def x_bound_rule(model, j):
        return (aml.value(model.l[j]), 10.0)

    model.x = aml.Var(model.N, bounds=x_bound_rule, initialize=6.0)

    model.obj = aml.Objective(
        expr=10.0 * model.x[1] * model.x[4] ** 2 * model.x[7] ** model.a / (model.x[2] * model.x[6] ** 3) + 15.0 *
        model.x[3] * model.x[4] / (model.x[1] * model.x[2] ** 2 *
        model.x[5] * model.x[7] ** 0.5) + 20.0 * model.x[2] * model.x[6] / (model.x[1] ** 2 * model.x[4] *
        model.x[5] ** 2) + 25.0 * model.x[1] ** 2 * model.x[2] ** 2 * model.x[5] ** 0.5 * model.x[7] / (
        model.x[3] * model.x[6] ** 2))

    model.c1 = aml.Constraint(
        expr=1.0 - .5 * model.x[1] ** 0.5 * model.x[7] / (model.x[3] * model.x[6] ** 2) - .7 * model.x[1] ** 3 *
        model.x[2] * model.x[6] * model.x[7] ** .5 / model.x[3] ** 2 \
        - .2 * model.x[3] * model.x[6] ** (2.0 / 3.0) * model.x[7] ** .25 / (model.x[2] * model.x[4] ** .5) >= 0)

    model.c2 = aml.Constraint(
        expr=1.0 - 1.3 * model.x[2] * model.x[6] / (model.x[1] ** .5 * model.x[3] * model.x[5]) - .8 * model.x[3] *
        model.x[6] ** 2 / (model.x[4] * model.x[5])
        - 3.1 * model.x[2] ** .5 * model.x[6] ** (1.0 / 3.0) / (model.x[1] * model.x[4] ** 2 * model.x[5]) >= 0)

    model.c3 = aml.Constraint(
        expr=1.0 - 2.0 * model.x[1] * model.x[5] * model.x[7] ** (1.0 / 3.0) / (model.x[3] ** 1.5 * model.x[6]) - .1 *
        model.x[2] * model.x[5] / (model.x[3] ** .5 * model.x[6] * model.x[7] ** .5) - model.x[2] * model.x[3] ** .5 * model.x[5] / model.x[1] - \
        .65 * model.x[3] * model.x[5] * model.x[7] / (model.x[2] ** 2 * model.x[6]) >= 0)

    model.c4 = aml.Constraint(expr=1.0 - .2 * model.x[2] * model.x[5] ** .5 * model.x[7] ** (1.0 / 3.0) / (model.x[1] ** 2 * model.x[4]) - .3 * model.x[1] ** .5 * model.x[2] ** 2 \
                                    * model.x[3] * model.x[4] ** (1.0 / 3.0) * model.x[7] ** .25 / model.x[5] ** (
                                    2.0 / 3.0) - .4 * model.x[3] * model.x[5] * model.x[7] ** .75 / \
                                    (model.x[1] ** 3 * model.x[2] ** 2) - .5 * model.x[4] * model.x[7] ** .5 / model.x[3] ** 2 >= 0)

    model.c5 = aml.Constraint(
        expr=10.0 * model.x[1] * model.x[4] ** 2 * model.x[7] ** model.a / (model.x[2] * model.x[6] ** 3) + 15 *
        model.x[3] * model.x[4] / (model.x[1] * model.x[2] ** 2 * model.x[5] *
        model.x[7] ** 0.5) + 20 *
        model.x[2] *
        model.x[6] / (
        model.x[1] ** 2 *
        model.x[4] *
        model.x[5] ** 2) \
        + 25.0 * model.x[1] ** 2 * model.x[2] ** 2 * model.x[5] ** 0.5 * model.x[7] / (
        model.x[3] * model.x[6] ** 2) >= 100.0)

    model.c6 = aml.Constraint(
        expr=10 * model.x[1] * model.x[4] ** 2 * model.x[7] ** model.a / (model.x[2] * model.x[6] ** 3) + 15 * model.x[3]
        * model.x[4] / (model.x[1] * model.x[2] ** 2 * model.x[5] *
        model.x[7] ** 0.5) + 20 * model.x[2] *
        model.x[6] / ( model.x[1] ** 2 * model.x[4] *model.x[5] ** 2)
        + 25.0 * model.x[1] ** 2 * model.x[2] ** 2 * model.x[5] ** 0.5 * model.x[7] / (
        model.x[3] * model.x[6] ** 2) <= 3000.0)

    return model


def create_model197():

    # hs106
    model = aml.ConcreteModel()
    model._name = 'hs106'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    N = 8
    model.I = RangeSet(1, N)
    model.M = RangeSet(2, 3)
    model.L = RangeSet(4, 8)
    a = 0.0025
    b = 0.01
    c = 833.3325
    d = 100.0
    e = 83333.33
    f = 1250.0
    g = 1250000.0
    h = 2500.0

    model.x = Var(model.I)
    model.x[1] = 5000.0
    model.x[2] = 5000.0
    model.x[3] = 5000.0
    model.x[4] = 200.0
    model.x[5] = 350.0
    model.x[6] = 150.0
    model.x[7] = 225.0
    model.x[8] = 425.0

    model.obj = aml.Objective(expr=model.x[1] + model.x[2] + model.x[3])

    model.c1 = Constraint(expr=1 - a * (model.x[4] + model.x[6]) >= 0)
    model.c2 = Constraint(expr=1 - a * (model.x[5] + model.x[7] - model.x[4]) >= 0)
    model.c3 = Constraint(expr=1 - b * (model.x[8] - model.x[5]) >= 0)
    model.c4 = Constraint(expr=model.x[1] * model.x[6] - c * model.x[4] - d * model.x[1] + e >= 0)
    model.c5 = Constraint(expr=model.x[2] * model.x[7] - f * model.x[5] - model.x[2] * model.x[4] + f * model.x[4] >= 0)
    model.c6 = Constraint(expr=model.x[3] * model.x[8] - g - model.x[3] * model.x[5] + h * model.x[5] >= 0)
    model.c7 = Constraint(expr=100 <= model.x[1] <= 10000)

    def cons8(model, i):
        return 1000 <= model.x[i] <= 10000

    model.c8 = Constraint(model.M, rule=cons8)

    def cons9(model, i):
        return 10 <= model.x[i] <= 1000

    model.c9 = Constraint(model.L, rule=cons9)

    return model


# ToDo: fails try with line search
def create_model198():

    # hs107
    model = aml.ConcreteModel()
    model._name = 'hs107'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    c = (48.4 / 50.176) * aml.sin(.25)
    d = (48.4 / 50.176) * aml.cos(.25)
    model.N = RangeSet(1, 9)
    model.x = Var(model.N)
    model.x[1] = 0.8
    model.x[2] = 0.8
    model.x[3] = 0.2
    model.x[4] = 0.2
    model.x[5] = 1.0454
    model.x[6] = 1.0454
    model.x[7] = 0.0
    model.x[8] = 0.0

    def _y1(model):
        return aml.sin(model.x[8])

    model.y1 = Expression(rule=_y1)

    def _y2(model):
        return aml.cos(model.x[8])

    model.y2 = Expression(rule=_y2)

    def _y3(model):
        return aml.sin(model.x[9])

    model.y3 = Expression(rule=_y3)

    def _y4(model):
        return aml.cos(model.x[9])

    model.y4 = Expression(rule=_y4)

    def _y5(model):
        return aml.sin(model.x[8] - model.x[9])

    model.y5 = Expression(rule=_y5)

    def _y6(model):
        return aml.cos(model.x[8] - model.x[9])

    model.y6 = Expression(rule=_y6)

    model.obj = aml.Objective(
        expr=3000 * model.x[1] + 1000 * model.x[1] ** 3 + 2000 * model.x[2] + 666.667 * model.x[2] ** 3)

    model.c1 = Constraint(
        expr=0.4 - model.x[1] + 2 * c * model.x[5] ** 2 - model.x[5] * model.x[6] \
        * (d * model.y1 + c * model.y2) - model.x[5] * model.x[7] * (d * model.y3 + c * model.y4) == 0)
    model.c2 = Constraint(expr=0.4 - model.x[2] + 2 * c * model.x[6] ** 2 + model.x[5] * model.x[6] \
    * (d * model.y1 - c * model.y2) + model.x[6] *model.x[7] * (d * model.y5 - c * model.y6) == 0)
    model.c3 = Constraint(expr=0.8 + 2 * c * model.x[7] ** 2 + model.x[5] * model.x[7] * (d * model.y3 \
    - c * model.y4) - model.x[6] * model.x[7] * (d * model.y5 + c * model.y6) == 0)
    model.c4 = Constraint(expr=0.2 - model.x[3] + 2 * d * model.x[5] ** 2 + model.x[5] * model.x[6] \
    * (c * model.y1 - d * model.y2) + model.x[5] * model.x[7] * (c * model.y3 - d * model.y4) == 0)
    model.c5 = Constraint(expr=0.2 - model.x[4] + 2 * d * model.x[6] ** 2 - model.x[5] * model.x[6] \
    * (c * model.y1 + d * model.y2) - model.x[6] * model.x[7] * (c * model.y5 + d * model.y6) == 0)

    return model


def create_model199():

    # hs109
    model = aml.ConcreteModel()
    model._name = 'hs109'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression

    model.N = RangeSet(1, 9)
    model.x = Var(model.N, initialize=1.0)

    model.obj = aml.Objective(expr=-.5 * (
    model.x[1] * model.x[4] - model.x[2] * model.x[3] + model.x[3] * model.x[9] - model.x[5] * model.x[9] + model.x[5] *
    model.x[8] - model.x[6] * model.x[7]))

    model.c1 = Constraint(expr=1 - model.x[3] ** 2 - model.x[4] ** 2 >= 0)
    model.c2 = Constraint(expr=1 - model.x[5] ** 2 - model.x[6] ** 2 >= 0)
    model.c3 = Constraint(expr=1 - model.x[9] ** 2 >= 0)
    model.c4 = Constraint(expr=1 - model.x[1] ** 2 - (model.x[2] - model.x[9]) ** 2 >= 0)
    model.c5 = Constraint(expr=1 - (model.x[1] - model.x[5]) ** 2 - (model.x[2] - model.x[6]) ** 2 >= 0)
    model.c6 = Constraint(expr=1 - (model.x[1] - model.x[7]) ** 2 - (model.x[2] - model.x[8]) ** 2 >= 0)
    model.c7 = Constraint(expr=1 - (model.x[3] - model.x[7]) ** 2 - (model.x[4] - model.x[8]) ** 2 >= 0)
    model.c8 = Constraint(expr=1 - (model.x[3] - model.x[5]) ** 2 - (model.x[4] - model.x[6]) ** 2 >= 0)
    model.c9 = Constraint(expr=1 - model.x[7] ** 2 - (model.x[8] - model.x[9]) ** 2 >= 0)
    model.c10 = Constraint(expr=model.x[1] * model.x[4] - model.x[2] * model.x[3] >= 0)
    model.c11 = Constraint(expr=model.x[3] * model.x[9] >= 0)
    model.c12 = Constraint(expr=-model.x[5] * model.x[9] >= 0)
    model.c13 = Constraint(expr=model.x[5] * model.x[8] - model.x[6] * model.x[7] >= 0)
    model.c14 = Constraint(expr=model.x[9] >= 0)

    return model


def create_model200():
    model = aml.ConcreteModel()
    model._name = 'hs110'

    Param = aml.Param
    RangeSet = aml.RangeSet
    Var = aml.Var
    value = aml.value
    Constraint = aml.Constraint
    Expression = aml.Expression
    sin = aml.sin
    cos = aml.cos
    Objective = aml.Objective

    model.N = RangeSet(1, 9)
    model.M = RangeSet(1, 2)
    model.L = RangeSet(3, 9)

    a = 50.176
    b1 = .25
    b = sin(b1)
    c = cos(b1)

    l_init = {}
    l_init[1] = 0
    l_init[2] = 0
    l_init[3] = -0.55
    l_init[4] = -0.55
    l_init[5] = 196.0
    l_init[6] = 196.0
    l_init[7] = 196.0
    l_init[8] = -400.0
    l_init[9] = -400.0

    u_init = {}
    u_init[1] = 0
    u_init[2] = 0
    u_init[3] = 0.55
    u_init[4] = 0.55
    u_init[5] = 252.0
    u_init[6] = 252.0
    u_init[7] = 252.0
    u_init[8] = 800.0
    u_init[9] = 800.0

    def l_init_rule(model, j):
        return l_init[j]

    def u_init_rule(model, j):
        return u_init[j]

    model.l = Param(model.N, initialize=l_init_rule)
    model.u = Param(model.N, initialize=u_init_rule)

    def x_bound_rule(model, j):
        if j in model.M:
            return (0, None)
        elif j in model.L:
            return (value(model.l[j]), value(model.u[j]))

    model.x = Var(model.N, bounds=x_bound_rule, initialize=0.0)

    def obj_rule(model):
        return 3 * model.x[1] + 1e-6 * model.x[1] ** 3 + 2 * model.x[2] + .522074e-6 * model.x[2] ** 3

    model.obj = Objective(rule=obj_rule)

    def cons1_rule(model):
        return model.x[4] - model.x[3] + .55 >= 0

    model.C1 = Constraint(rule=cons1_rule)

    def cons2_rule(model):
        return model.x[3] - model.x[4] + .55 >= 0

    model.C2 = Constraint(rule=cons2_rule)

    def cons3_rule(model):
        return 2250000 - model.x[1] ** 2 - model.x[8] ** 2 >= 0

    model.C3 = Constraint(rule=cons3_rule)

    def cons4_rule(model):
        return 2250000 - model.x[2] ** 2 - model.x[9] ** 2 >= 0

    model.C4 = Constraint(rule=cons4_rule)

    def cons5_rule(model):
        return model.x[5] * model.x[6] * sin(-model.x[3] - .25) + model.x[5] * model.x[7] * sin(
            -model.x[4] - .25) + 2 * b * model.x[5] ** 2 - a * model.x[1] + 400 * a == 0

    model.C5 = Constraint(rule=cons5_rule)

    def cons6_rule(model):
        return model.x[5] * model.x[6] * sin(model.x[3] - .25) + model.x[6] * model.x[7] * sin(
            model.x[3] - model.x[4] - .25) + 2 * b * model.x[6] ** 2 - a * model.x[2] + 400 * a == 0

    model.C6 = Constraint(rule=cons6_rule)

    def cons7_rule(model):
        return model.x[5] * model.x[7] * sin(model.x[4] - .25) + model.x[6] * model.x[7] * sin(
            model.x[4] - model.x[3] - .25) + 2 * b * model.x[7] ** 2 + 881.779 * a == 0

    model.C7 = Constraint(rule=cons7_rule)

    def cons8_rule(model):
        return a * model.x[8] + model.x[5] * model.x[6] * cos(-model.x[3] - .25) + model.x[5] * model.x[7] * cos(
            -model.x[4] - .25) - 200 * a - 2 * c * model.x[5] ** 2 + .7533e-3 * a * model.x[5] ** 2 == 0

    model.C8 = Constraint(rule=cons8_rule)

    def cons9_rule(model):
        return a * model.x[9] + model.x[5] * model.x[6] * cos(model.x[3] - .25) + model.x[6] * model.x[7] * cos(
            model.x[3] - model.x[4] - .25) - 2 * c * model.x[6] ** 2 + .7533e-3 * a * model.x[6] ** 2 - 200 * a == 0

    model.C9 = Constraint(rule=cons9_rule)

    def cons10_rule(model):
        return model.x[5] * model.x[7] * cos(model.x[4] - .25) + model.x[6] * model.x[7] * cos(
            model.x[4] - model.x[3] - .25) - 2 * c * model.x[7] ** 2 + 22.938 * a + .7533e-3 * a * model.x[7] ** 2 == 0

    model.C10 = Constraint(rule=cons10_rule)

    def cons11_rule(model, i):
        return model.x[i] >= 0

    # model.C11 = Constraint([1,2], rule=cons11_rule)

    def cons12_rule(model, i):
        return -.55 <= model.x[i] <= .55

    # model.C12 = Constraint([3,4], rule=cons12_rule)

    def cons13_rule(model, i):
        return 196 <= model.x[i] <= 252

    # model.C13 = Constraint([5,6,7], rule=cons13_rule)

    def cons14_rule(model, i):
        return -400 <= model.x[i] <= 800
    # model.C14 = Constraint([8,9], rule=cons14_rule)
    return model


# ToDo: pass but try with line search
def create_model201():

    # ksip
    model = aml.ConcreteModel()
    model._name = 'ksip'
    n = 20.0
    m = 1000.0
    model.N = aml.RangeSet(1, n)
    model.M = aml.RangeSet(0, m)
    model.x = aml.Var(model.N, initialize=2.0)

    model.obj = aml.Objective(expr=sum((model.x[j] ** 2 / (2.0 * j) + model.x[j] / float(j)) for j in model.N))

    def cons1_rule(model, i):
        return sum((i / m) ** (j - 1) * model.x[j] for j in model.N) >= aml.sin(i / m)

    model.c = aml.Constraint(model.M, rule=cons1_rule)
    return model