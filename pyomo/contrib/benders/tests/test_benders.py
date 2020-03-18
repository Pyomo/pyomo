import pyutilib.th as unittest
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pe
try:
    import mpi4py
    mpi4py_available = True
except:
    mpi4py_available = False
try:
    import numpy as np
    numpy_available = True
except:
    numpy_available = False


class TestBenders(unittest.TestCase):
    @unittest.skipIf(not mpi4py_available, 'mpi4py is not available.')
    @unittest.skipIf(not numpy_available, 'numpy is not available.')
    def test_grothey(self):
        def create_master():
            m = pe.ConcreteModel()
            m.y = pe.Var(bounds=(1, None))
            m.eta = pe.Var(bounds=(-10, None))
            m.obj = pe.Objective(expr=m.y ** 2 + m.eta)
            return m

        def create_subproblem(master):
            m = pe.ConcreteModel()
            m.x1 = pe.Var()
            m.x2 = pe.Var()
            m.y = pe.Var()
            m.obj = pe.Objective(expr=-m.x2)
            m.c1 = pe.Constraint(expr=(m.x1 - 1) ** 2 + m.x2 ** 2 <= pe.log(m.y))
            m.c2 = pe.Constraint(expr=(m.x1 + 1) ** 2 + m.x2 ** 2 <= pe.log(m.y))

            complicating_vars_map = pe.ComponentMap()
            complicating_vars_map[master.y] = m.y

            return m, complicating_vars_map

        m = create_master()
        master_vars = [m.y]
        m.benders = BendersCutGenerator()
        m.benders.set_input(master_vars=master_vars, tol=1e-8)
        m.benders.add_subproblem(subproblem_fn=create_subproblem,
                                 subproblem_fn_kwargs={'master': m},
                                 master_eta=m.eta,
                                 subproblem_solver='ipopt', )
        opt = pe.SolverFactory('ipopt')

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break
        self.assertAlmostEqual(m.y.value, 2.721381, 4)
        self.assertAlmostEqual(m.eta.value, -0.0337568, 4)

    @unittest.skipIf(not mpi4py_available, 'mpi4py is not available.')
    @unittest.skipIf(not numpy_available, 'numpy is not available.')
    def test_farmer(self):
        class Farmer(object):
            def __init__(self):
                self.crops = ['WHEAT', 'CORN', 'SUGAR_BEETS']
                self.total_acreage = 500
                self.PriceQuota = {'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0}
                self.SubQuotaSellingPrice = {'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0}
                self.SuperQuotaSellingPrice = {'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0}
                self.CattleFeedRequirement = {'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0}
                self.PurchasePrice = {'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0}
                self.PlantingCostPerAcre = {'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0}
                self.scenarios = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
                self.crop_yield = dict()
                self.crop_yield['BelowAverageScenario'] = {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0}
                self.crop_yield['AverageScenario'] = {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0}
                self.crop_yield['AboveAverageScenario'] = {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}
                self.scenario_probabilities = dict()
                self.scenario_probabilities['BelowAverageScenario'] = 0.3333
                self.scenario_probabilities['AverageScenario'] = 0.3334
                self.scenario_probabilities['AboveAverageScenario'] = 0.3333

        def create_master(farmer):
            m = pe.ConcreteModel()

            m.crops = pe.Set(initialize=farmer.crops, ordered=True)
            m.scenarios = pe.Set(initialize=farmer.scenarios, ordered=True)

            m.devoted_acreage = pe.Var(m.crops, bounds=(0, farmer.total_acreage))
            m.eta = pe.Var(m.scenarios)
            for s in m.scenarios:
                m.eta[s].setlb(-432000 * farmer.scenario_probabilities[s])

            m.total_acreage_con = pe.Constraint(expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage)

            m.obj = pe.Objective(
                expr=sum(farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop] for crop in m.crops) + sum(
                    m.eta.values()))
            return m

        def create_subproblem(master, farmer, scenario):
            m = pe.ConcreteModel()

            m.crops = pe.Set(initialize=farmer.crops, ordered=True)

            m.devoted_acreage = pe.Var(m.crops)
            m.QuantitySubQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
            m.QuantitySuperQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
            m.QuantityPurchased = pe.Var(m.crops, bounds=(0.0, None))

            def EnforceCattleFeedRequirement_rule(m, i):
                return (farmer.CattleFeedRequirement[i] <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) +
                        m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i])

            m.EnforceCattleFeedRequirement = pe.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

            def LimitAmountSold_rule(m, i):
                return m.QuantitySubQuotaSold[i] + m.QuantitySuperQuotaSold[i] - (
                            farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) <= 0.0

            m.LimitAmountSold = pe.Constraint(m.crops, rule=LimitAmountSold_rule)

            def EnforceQuotas_rule(m, i):
                return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])

            m.EnforceQuotas = pe.Constraint(m.crops, rule=EnforceQuotas_rule)

            obj_expr = sum(farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops)
            obj_expr -= sum(farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops)
            obj_expr -= sum(farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops)
            m.obj = pe.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)

            complicating_vars_map = pe.ComponentMap()
            for crop in m.crops:
                complicating_vars_map[master.devoted_acreage[crop]] = m.devoted_acreage[crop]

            return m, complicating_vars_map

        farmer = Farmer()
        m = create_master(farmer=farmer)
        master_vars = list(m.devoted_acreage.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(master_vars=master_vars, tol=1e-8)
        for s in farmer.scenarios:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs['master'] = m
            subproblem_fn_kwargs['farmer'] = farmer
            subproblem_fn_kwargs['scenario'] = s
            m.benders.add_subproblem(subproblem_fn=create_subproblem,
                                     subproblem_fn_kwargs=subproblem_fn_kwargs,
                                     master_eta=m.eta[s],
                                     subproblem_solver='glpk')
        opt = pe.SolverFactory('glpk')

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break

        self.assertAlmostEqual(m.devoted_acreage['CORN'].value, 80, 7)
        self.assertAlmostEqual(m.devoted_acreage['SUGAR_BEETS'].value, 250, 7)
        self.assertAlmostEqual(m.devoted_acreage['WHEAT'].value, 170, 7)
