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
    model = aml.ConcreteModel()
    model._name = 'model10'
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


def create_model11():

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


def create_model12():

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


def create_model13():

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


def create_model14():

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


def create_model15():

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


def create_model16():

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


def create_model17():

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


def create_model18():

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


def create_model19():

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


def create_model20():

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
