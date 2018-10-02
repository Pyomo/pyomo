import pyutilib.th as unittest
try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.interfaces import PyomoNLP, AmplNLP
from pyomo.contrib.pynumero.algorithms import basic_sqp
import pyomo.environ as pe
import pyomo.dae as dae


@unittest.skip
class TestSQP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.ipopt = pe.SolverFactory('ipopt')
        cls.ipopt.options['nlp_scaling_method'] = 'none'
        cls.ipopt.options['linear_system_scaling'] = 'none'

        cls.pynumero_sqp = pe.SolverFactory('pynumero_sqp')

    def _compare_solutions(self, model1, model2, places=7):

        # solve with ipopt
        self.ipopt.solve(model1)

        # solve with pynumero
        self.pynumero_sqp.solve(model2)

        vars = model1.component_map(pe.Var, active=True)
        for k, v in vars.items():
            if not isinstance(v, dae.DerivativeVar):
                cuid = pe.ComponentUID(v)
                v1 = cuid.find_component_on(model2)
                if not v.is_indexed():
                    self.assertAlmostEqual(pe.value(v), pe.value(v1), places)
                else:
                    for j in v.keys():
                        self.assertAlmostEqual(pe.value(v[j]), pe.value(v1[j]), places)

    def test_rosenbrock(self):

        model = pe.ConcreteModel()
        model.n_vars = 10
        model.lb = -10
        model.ub = 20
        model.init = 2.0
        model.index_vars = range(model.n_vars)
        model.x = pe.Var(model.index_vars, initialize=model.init, bounds=(model.lb, model.ub))

        def rule_constraint(m, i):
            return m.x[i] ** 3 - 1.0 == 0

        model.c = pe.Constraint(model.index_vars, rule=rule_constraint)

        model.obj = pe.Objective(expr=sum((model.x[i] - 2) ** 2 for i in model.index_vars))

        instance1 = model.clone()

        self._compare_solutions(instance1, model)

    def test_basic1(self):

        m = pe.ConcreteModel()
        m.x = pe.Var([1, 2], initialize=1.0)
        m.c1 = pe.Constraint(expr=m.x[1] + m.x[2] - 1 == 0)
        m.obj = pe.Objective(expr=2 * m.x[1] ** 2 + m.x[2] ** 2)

        model = m.clone()
        self._compare_solutions(m, model)

    def test_basic2(self):

        m = pe.ConcreteModel()
        m.x = pe.Var([1, 2], initialize=1.0)
        m.c1 = pe.Constraint(expr=4 * m.x[1] ** 2 + m.x[2] ** 2 - 8 == 0)
        m.obj = pe.Objective(expr=-2 * m.x[1] + m.x[2])

        model = m.clone()
        self._compare_solutions(m, model)

    def test_basic3(self):

        m = pe.ConcreteModel()
        m.x = pe.Var([1, 2, 3], initialize=5.0)
        m.c1 = pe.Constraint(expr=m.x[1] - 1 == 0)
        m.c2 = pe.Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 - 1.0 == 0)
        m.obj = pe.Objective(expr=m.x[1] + m.x[2] + m.x[3] ** 2)

        model = m.clone()
        self._compare_solutions(m, model)

    def test_basic4(self):

        m = pe.ConcreteModel()
        m.x = pe.Var([1, 2, 3], initialize=1.0)
        m.c1 = pe.Constraint(expr=m.x[1] + m.x[2] + 1.5 * m.x[3] == 1.2)
        m.c2 = pe.Constraint(expr=m.x[1] + m.x[2] + m.x[3] == 1.0)
        m.obj = pe.Objective(
            expr=400 * m.x[1] ** 2 + 800 * m.x[2] ** 2 + 200 * m.x[1] * m.x[2] + 1600 * m.x[3] ** 2 + 400 * m.x[2] *
                                                                                                      m.x[3])
        model = m.clone()
        self._compare_solutions(m, model)

    def test_aircrafta_cute(self):

        model = pe.ConcreteModel()

        model.rollrate = pe.Var(initialize=0.0)
        model.pitchrat = pe.Var(initialize=0.0)
        model.yawrate = pe.Var(initialize=0.0)
        model.attckang = pe.Var(initialize=0.0)
        model.sslipang = pe.Var(initialize=0.0)
        model.elevator = pe.Var(initialize=0.0)
        model.aileron = pe.Var(initialize=0.0)
        model.rudderdf = pe.Var(initialize=0.0)

        model.elevator = 0.1
        model.elevator.fixed = True
        model.aileron = 0
        model.aileron.fixed = True
        model.rudderdf = 0
        model.rudderdf.fixed = True

        def f(model):
            return 0

        model.f = pe.Objective(rule=f, sense=pe.minimize)

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

        model.cons1 = pe.Constraint(rule=cons1)
        model.cons2 = pe.Constraint(rule=cons2)
        model.cons3 = pe.Constraint(rule=cons3)
        model.cons4 = pe.Constraint(rule=cons4)
        model.cons5 = pe.Constraint(rule=cons5)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_argtrig_cute(self):

        model = pe.ConcreteModel()

        N = 100
        model.x = pe.Var(pe.RangeSet(1, N), initialize=1.0 / N)

        def f_rule(model):
            return 0

        model.f = pe.Objective(rule=f_rule)

        def cons1_rule(model, i):
            return i * (pe.cos(model.x[i]) + pe.sin(model.x[i])) + sum(pe.cos(model.x[j]) for j in range(1, N + 1)) - (
            N + i) == 0

        model.cons = pe.Constraint(pe.RangeSet(1, N), rule=cons1_rule)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_bdvalue_cute(self):

        model = pe.ConcreteModel()

        model.ndp = 5002
        model.h = 1.0 / (model.ndp - 1)
        model.S = pe.RangeSet(1, model.ndp)
        model.SS = pe.RangeSet(2, model.ndp - 1)

        def x_init(model, i):
            return ((i - 1) * model.h) * ((i - 1) * model.h - 1)

        model.x = pe.Var(model.S, initialize=x_init)

        model.x[1] = 0
        model.x[1].fixed = True
        model.x[model.ndp].fixed = True

        def f(model):
            return 0

        model.f = pe.Objective(rule=f)

        def cons(model, i):
            return (-model.x[i - 1] + 2 * model.x[i] - model.x[i + 1] + 0.5 * model.h ** 2 * (
            model.x[i] + i * model.h + 1) ** 3) == 0

        model.cons = pe.Constraint(model.SS, rule=cons)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_biggs3_cute(self):

        model = pe.ConcreteModel()

        model.N = pe.Param(initialize=6)
        model.M = pe.Param(initialize=13)
        model.S = pe.RangeSet(1, model.N)
        model.SS = pe.RangeSet(1, model.M)

        model.x = pe.Var(model.S)
        model.x[1] = 1.0
        model.x[2] = 2.0
        model.x[3] = 1.0
        model.x[4] = 1.0
        model.x[5] = 4.0
        model.x[6] = 3.0

        def f_rule(model):
            sum1 = 0.0
            for i in model.SS:
                sum1 += (-pe.exp(-0.1 * i) + 5 * pe.exp(-i) - 3 * pe.exp(-0.4 * i) + model.x[3] * pe.exp(-0.1 * i * model.x[1]) \
                         - model.x[4] * pe.exp(-0.1 * i * model.x[2]) + model.x[6] * pe.exp(-0.1 * i * model.x[5])) ** 2
            return sum1

        model.f = pe.Objective(rule=f_rule)

        def cons1(model):
            return model.x[3] == 1

        model.cons1 = pe.Constraint(rule=cons1)

        def cons2(model):
            return model.x[5] == 4

        model.cons2 = pe.Constraint(rule=cons2)

        def cons3(model):
            return model.x[6] == 3

        model.cons3 = pe.Constraint(rule=cons3)

        model2 = model.clone()
        self._compare_solutions(model, model2, places=5)

    def test_booth_cute(self):

        model = pe.ConcreteModel()

        model.Sx = pe.RangeSet(1, 2)
        model.x = pe.Var(model.Sx)

        def f(model):
            return 0

        model.f = pe.Objective(rule=f, sense=pe.minimize)

        def cons1(model):
            return (0, model.x[1] + 2 * model.x[2] - 7)

        model.cons1 = pe.Constraint(rule=cons1)

        def cons2(model):
            return (0, 2 * model.x[1] + model.x[2] - 5)

        model.cons2 = pe.Constraint(rule=cons2)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_broydn3d_cute(self):

        model = pe.ConcreteModel()

        N = 10000
        kappa1 = 2.0
        kappa2 = 1.0

        model.x = pe.Var(pe.RangeSet(1, N), initialize=-1.0)

        def f_rule(model):
            return 0

        model.f = pe.Objective(rule=f_rule)

        def con1(model):
            return (-2 * model.x[2] + kappa2 + (3 - kappa1 * model.x[1]) * model.x[1]) == 0

        model.cons1 = pe.Constraint(rule=con1)

        def con2(model, i):
            return (-model.x[i - 1] - 2 * model.x[i + 1] + kappa2 + (3 - kappa1 * model.x[i]) * model.x[i]) == 0

        model.cons2 = pe.Constraint(pe.RangeSet(2, N - 1), rule=con2)

        def con3(model):
            return (-model.x[N - 1] + kappa2 + (3 - kappa1 * model.x[N]) * model.x[N]) == 0

        model.cons3 = pe.Constraint(rule=con3)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_b9_cute(self):

        model = pe.ConcreteModel()

        model.S = pe.RangeSet(1, 4)
        model.x = pe.Var(model.S, initialize=2.0)

        def f(model):
            return -model.x[1]

        model.f = pe.Objective(rule=f, sense=pe.minimize)

        def cons1(model):
            return model.x[2] - model.x[1] ** 3 - model.x[3] ** 2 == 0

        model.cons1 = pe.Constraint(rule=cons1)

        def cons2(model):
            return -model.x[2] + model.x[1] ** 2 - model.x[4] ** 2 == 0

        model.cons2 = pe.Constraint(rule=cons2)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_catena_cute(self):

        model = pe.ConcreteModel()

        model.N = pe.Param(initialize=10)

        model.gamma = pe.Param(initialize=9.81)
        model.tmass = pe.Param(initialize=500.0)
        model.bl = pe.Param(initialize=1.0)
        model.fract = pe.Param(initialize=0.6)

        def len_rule(model):
            return pe.value(model.bl) * (pe.value(model.N) + 1) * pe.value(model.fract)

        def mass_rule(model):
            return pe.value(model.tmass) / (pe.value(model.N) + 1.0)

        def mg_rule(model):
            return pe.value(model.mass) * pe.value(model.gamma)

        model.length = pe.Param(initialize=len_rule)
        model.mass = pe.Param(initialize=mass_rule)
        model.mg = pe.Param(initialize=mg_rule)

        model.Sv = pe.RangeSet(0, model.N + 1)
        model.So = pe.RangeSet(1, model.N)
        model.Sc = pe.RangeSet(1, model.N + 1)

        def x_rule(model, i):
            return i * pe.value(model.length) / (pe.value(model.N) + 1.0)

        def y_rule(model, i):
            return -i * pe.value(model.length) / (pe.value(model.N) + 1.0)

        # def fix1(model, i):
        #       if i == 0:
        #               return (0,0)
        #       else:
        #               return (None,None)
        # def fix2(model, i):
        #       if i == model.N+1:
        #               return (model.length,modell.ength)
        #       if i == 0:
        #               return (0,0)
        #       else:
        #               return (None,None)

        model.x = pe.Var(model.Sv, initialize=x_rule)
        model.y = pe.Var(model.Sv, initialize=y_rule)
        model.z = pe.Var(model.Sv, initialize=0.0)

        def f(model):
            obsum = 0
            for i in model.So:
                obsum += pe.value(model.mg) * model.y[i]
            obsum += pe.value(model.mg) * model.y[pe.value(model.N) + 1] / 2.0
            expr = pe.value(model.mg) * model.y[0] / 2.0 + obsum
            return expr

        model.f = pe.Objective(rule=f, sense=pe.minimize)

        def cons1(model, i):
            expr = (model.x[i] - model.x[i - 1]) ** 2 + (model.y[i] - model.y[i - 1]) ** 2 + (model.z[i] - model.z[
                i - 1]) ** 2
            return expr == pe.value(model.bl) ** 2

        model.cons1 = pe.Constraint(model.Sc, rule=cons1)

        model.x[0] = 0.0
        model.x[0].fixed = True
        model.y[0] = 0.0
        model.y[0].fixed = True
        model.z[0] = 0.0
        model.z[0].fixed = True
        model.x[pe.value(model.N) + 1] = pe.value(model.length)
        model.x[pe.value(model.N) + 1].fixed = True

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_cbratu2d_cute(self):

        model = pe.ConcreteModel()

        p = 23
        l = 5.0
        h = 1.0 / (p - 1.0)
        c = h ** 2 / l

        model.u = pe.Var(pe.RangeSet(1, p), pe.RangeSet(1, p), initialize=0.0)
        model.x = pe.Var(pe.RangeSet(1, p), pe.RangeSet(1, p), initialize=0.0)

        model.f = pe.Objective(expr=0.0)

        def con1(model, i, j):
            return (4 * model.u[i, j] - model.u[i + 1, j] - model.u[i - 1, j] - model.u[i, j + 1] - \
                    model.u[i, j - 1] - c * pe.exp(model.u[i, j]) * pe.cos(model.x[i, j])) == 0

        model.cons1 = pe.Constraint(pe.RangeSet(2, p - 1), pe.RangeSet(2, p - 1), rule=con1)

        def con2(model, i, j):
            return (4 * model.x[i, j] - model.x[i + 1, j] - model.x[i - 1, j] - model.x[i, j + 1] - \
                    model.x[i, j - 1] - c * pe.exp(model.u[i, j]) * pe.sin(model.x[i, j])) == 0

        model.cons2 = pe.Constraint(pe.RangeSet(2, p - 1), pe.RangeSet(2, p - 1), rule=con2)

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

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_genhs28_cute(self):

        model = pe.ConcreteModel()

        N = 10

        def x_init_rule(model, i):
            if i == 1:
                return -4.0
            else:
                return 1.0

        model.x = pe.Var(pe.RangeSet(1, N), initialize=x_init_rule)

        def f_rule(model):
            return sum((model.x[i] + model.x[i + 1]) ** 2 for i in range(1, N))

        model.f = pe.Objective(rule=f_rule)

        def cons_rule(model, i):
            return -1.0 + model.x[i] + 2 * model.x[i + 1] + 3 * model.x[i + 2] == 0

        model.cons = pe.Constraint(pe.RangeSet(1, N - 2), rule=cons_rule)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_gottfr_cute(self):

        model = pe.ConcreteModel()

        model.x = pe.Var(pe.RangeSet(1, 2), initialize=0.5)

        def f_rule(model):
            return 0

        model.f = pe.Objective(rule=f_rule)

        def cons_rule(model):
            return (model.x[1] - 0.1136 * (model.x[1] + 3.0 * model.x[2]) * (1 - model.x[1])) == 0

        model.cons = pe.Constraint(rule=cons_rule)

        def cons2_rule(model):
            return (model.x[2] + 7.5 * (2.0 * model.x[1] - model.x[2]) * (1 - model.x[2])) == 0

        model.cons2 = pe.Constraint(rule=cons2_rule)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_hypcir_cute(self):

        model = pe.ConcreteModel()
        model.N = pe.RangeSet(1, 2)
        model.x = pe.Var(model.N)
        model.x[1] = 0.0
        model.x[2] = 1.0

        model.f = pe.Objective(expr=0)

        model.cons1 = pe.Constraint(expr=model.x[1] * model.x[2] - 1.0 == 0)
        model.cons2 = pe.Constraint(expr=model.x[1] ** 2 + model.x[2] ** 2 - 4.0 == 0)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_cvxqp3_cute(self):

        model = pe.ConcreteModel()
        N = 1000
        model.N = N
        model.M = int(3 * N / 4)

        model.S = range(1, model.N + 1)
        model.SS = range(1, model.M + 1)
        model.x = pe.Var(model.S, initialize=0.5)

        def f(model):
            return sum([(model.x[i] + model.x[((2 * i - 1) % pe.value(model.N)) + 1] + model.x[
                ((3 * i - 1) % pe.value(model.N)) + 1]) ** 2 * i / 2.0 for i in model.S])

        model.f = pe.Objective(rule=f, sense=pe.minimize)

        def cons1(model, i):
            return model.x[i] + 2 * model.x[((4 * i - 1) % pe.value(model.N)) + 1] + 3 * model.x[
                ((5 * i - 1) % pe.value(model.N)) + 1] - 6.0 == 0

        model.cons1 = pe.Constraint(model.SS, rule=cons1)

        model2 = model.clone()
        self._compare_solutions(model, model2)

    def test_distillation_model(self):

        initial_concentrations = dict()
        initial_concentrations[1] = 0.935419416
        initial_concentrations[2] = 0.900525537
        initial_concentrations[3] = 0.862296451
        initial_concentrations[4] = 0.821699403
        initial_concentrations[5] = 0.779990796
        initial_concentrations[6] = 0.738571686
        initial_concentrations[7] = 0.698804909
        initial_concentrations[8] = 0.661842534
        initial_concentrations[9] = 0.628507776
        initial_concentrations[10] = 0.5992527
        initial_concentrations[11] = 0.57418568
        initial_concentrations[12] = 0.553144227
        initial_concentrations[13] = 0.535784544
        initial_concentrations[14] = 0.52166551
        initial_concentrations[15] = 0.510314951
        initial_concentrations[16] = 0.501275092
        initial_concentrations[17] = 0.494128917
        initial_concentrations[18] = 0.48544992
        initial_concentrations[19] = 0.474202481
        initial_concentrations[20] = 0.459803499
        initial_concentrations[21] = 0.441642973
        initial_concentrations[22] = 0.419191098
        initial_concentrations[23] = 0.392055492
        initial_concentrations[24] = 0.360245926
        initial_concentrations[25] = 0.32407993
        initial_concentrations[26] = 0.284676816
        initial_concentrations[27] = 0.243209213
        initial_concentrations[28] = 0.201815683
        initial_concentrations[29] = 0.16177269
        initial_concentrations[30] = 0.12514971
        initial_concentrations[31] = 0.092458326
        initial_concentrations[32] = 0.064583177

        model = pe.ConcreteModel()

        # define sets
        model.trays = range(1, 33)
        model.rectification_trays = range(2, 17)
        model.stripping_trays = range(18, 32)
        model.t = dae.ContinuousSet(bounds=(0.0, 52.0))

        # define parameters
        model.Feed = pe.Param(initialize=0.4)
        model.x_Feed = pe.Param(initialize=0.5)
        model.D = pe.Param(initialize=model.x_Feed * model.Feed)
        model.vol = pe.Param(initialize=1.6)
        model.Atray = pe.Param(initialize=0.25)
        model.Acond = pe.Param(initialize=0.5)
        model.Areb = pe.Param(initialize=1.0)
        model.feed_tray = 17
        model.x0 = initial_concentrations

        # define variables

        # liquid concentration
        model.x = pe.Var(model.trays, model.t, initialize=lambda m, n, t: initial_concentrations[n])
        model.dx = dae.DerivativeVar(model.x)

        # vapor concentration
        model.y = pe.Var(model.trays, model.t)

        # reflux ratio
        model.rr = pe.Var(model.t, initialize=3.0)

        # feed flow rate
        model.FL = pe.Var(model.t, initialize=1)  # L2 in paper
        model.L = pe.Var(model.t, initialize=0.6)  # L1 in paper
        model.V = pe.Var(model.t, initialize=0.8)

        # model.u1 = pe.Var(model.t, initialize=3.0, bounds=(1, 5))
        model.u1 = pe.Var(model.t, initialize=3.0)

        model.alpha = pe.Param(initialize=1000)
        model.rho = pe.Param(initialize=1)
        model.u1_ref = pe.Param(initialize=2.0)
        model.y1_ref = pe.Param(initialize=0.895814)

        ###
        # Model constraints
        ###

        def vapor_column_rule(m, t):
            return m.V[t] == m.L[t] + m.D

        model.vapor_column = pe.Constraint(model.t, rule=vapor_column_rule)

        def flowrate_stripping_rule(m, t):
            return m.FL[t] == m.Feed + m.L[t]

        model.flowrate_stripping = pe.Constraint(model.t, rule=flowrate_stripping_rule)

        def reflux_ratio_rule(m, t):
            return m.rr[t] == m.u1[t]

        model.reflux_ratio = pe.Constraint(model.t, rule=reflux_ratio_rule)

        def flowrate_rectificaiton_rule(m, t):
            return m.L[t] == m.rr[t] * m.D

        model.flowrate_rectificaiton = pe.Constraint(model.t, rule=flowrate_rectificaiton_rule)

        def mole_frac_balance_rule(m, n, t):
            return m.y[n, t] == m.x[n, t] * m.vol / (1 + ((m.vol - 1) * m.x[n, t]))

        model.mole_frac_balance = pe.Constraint(model.trays, model.t, rule=mole_frac_balance_rule)

        # differential equations

        def first_tray_component_balance(m, t):
            if t == m.t.first():
                return pe.Constraint.Skip
            return m.dx[1, t] == 1 / m.Acond * m.V[t] * (m.y[2, t] - m.x[1, t])

        model.first_tray_component_balance = pe.Constraint(model.t,
                                                           rule=first_tray_component_balance)

        def rectification_component_balance(m, n, t):
            if t == m.t.first():
                return pe.Constraint.Skip
            return m.dx[n, t] == 1 / m.Atray * \
                                 (m.L[t] * (m.x[n - 1, t] - m.x[n, t]) - m.V[t] * (m.y[n, t] - m.y[n + 1, t]))

        model.rectification_component_balance = pe.Constraint(model.rectification_trays,
                                                              model.t,
                                                              rule=rectification_component_balance)

        def feed_component_balance(m, t):
            if t == m.t.first():
                return pe.Constraint.Skip
            n = m.feed_tray
            return m.dx[n, t] == 1.0 / m.Atray * \
                                 (m.Feed * m.x_Feed + m.L[t] * m.x[n - 1, t] - m.FL[t] * m.x[n, t] - m.V[t] * (
                                 m.y[n, t] - m.y[n + 1, t]))

        model.feed_component_balance = pe.Constraint(model.t,
                                                     rule=feed_component_balance)

        def stripping_component_balance(m, n, t):
            if t == m.t.first():
                return pe.Constraint.Skip
            return m.dx[n, t] == 1 / m.Atray * \
                                 (m.FL[t] * (m.x[n - 1, t] - m.x[n, t]) - m.V[t] * (m.y[n, t] - m.y[n + 1, t]))

        model.stripping_component_balance = pe.Constraint(model.stripping_trays,
                                                          model.t,
                                                          rule=stripping_component_balance)

        def last_tray_component_balance(m, t):
            if t == m.t.first():
                return pe.Constraint.Skip
            n = 32
            return m.dx[n, t] == 1 / m.Areb * (
                m.FL[t] * m.x[n - 1, t] - (m.Feed - m.D) * m.x[n, t] - m.V[t] * m.y[n, t])

        model.last_tray_component_balance = pe.Constraint(model.t,
                                                          rule=last_tray_component_balance)

        def _init_rule(m):
            t0 = m.t.first()
            for n in m.trays:
                yield m.x[n, t0] == m.x0[n]

        model.init_rule = pe.ConstraintList(rule=_init_rule)

        def _int_rule(m, t):
            return m.alpha * (m.y[1, t] - m.y1_ref) ** 2 + m.rho * (m.u1[t] - m.u1_ref) ** 2

        model.integral = dae.Integral(model.t, wrt=model.t, rule=_int_rule)
        model.OBJ = pe.Objective(expr=model.integral)

        discretizer = pe.TransformationFactory('dae.collocation')
        discretizer.apply_to(model, nfe=30, ncp=3)
        model2 = model.clone()

        self._compare_solutions(model, model2)

