#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
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


ipopt_opt = pyo.SolverFactory('ipopt')
ipopt_available = ipopt_opt.available(exception_flag=False)

cplex_opt = pyo.SolverFactory('cplex_direct')
cplex_available = cplex_opt.available(exception_flag=False)


@unittest.category('mpi')
class MPITestBenders(unittest.TestCase):
    @unittest.skipIf(not mpi4py_available, 'mpi4py is not available.')
    @unittest.skipIf(not numpy_available, 'numpy is not available.')
    @unittest.skipIf(not cplex_available, 'cplex is not available.')
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
            m = pyo.ConcreteModel()

            m.crops = pyo.Set(initialize=farmer.crops, ordered=True)
            m.scenarios = pyo.Set(initialize=farmer.scenarios, ordered=True)

            m.devoted_acreage = pyo.Var(m.crops, bounds=(0, farmer.total_acreage))
            m.eta = pyo.Var(m.scenarios)
            for s in m.scenarios:
                m.eta[s].setlb(-432000 * farmer.scenario_probabilities[s])

            m.total_acreage_con = pyo.Constraint(expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage)

            m.obj = pyo.Objective(
                expr=sum(farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop] for crop in m.crops) + sum(
                    m.eta.values()))
            return m

        def create_subproblem(master, farmer, scenario):
            m = pyo.ConcreteModel()

            m.crops = pyo.Set(initialize=farmer.crops, ordered=True)

            m.devoted_acreage = pyo.Var(m.crops)
            m.QuantitySubQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
            m.QuantitySuperQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
            m.QuantityPurchased = pyo.Var(m.crops, bounds=(0.0, None))

            def EnforceCattleFeedRequirement_rule(m, i):
                return (farmer.CattleFeedRequirement[i] <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) +
                        m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i])

            m.EnforceCattleFeedRequirement = pyo.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

            def LimitAmountSold_rule(m, i):
                return m.QuantitySubQuotaSold[i] + m.QuantitySuperQuotaSold[i] - (
                            farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) <= 0.0

            m.LimitAmountSold = pyo.Constraint(m.crops, rule=LimitAmountSold_rule)

            def EnforceQuotas_rule(m, i):
                return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])

            m.EnforceQuotas = pyo.Constraint(m.crops, rule=EnforceQuotas_rule)

            obj_expr = sum(farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops)
            obj_expr -= sum(farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops)
            obj_expr -= sum(farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops)
            m.obj = pyo.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)

            complicating_vars_map = pyo.ComponentMap()
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
                                     subproblem_solver='cplex_direct')
        opt = pyo.SolverFactory('cplex_direct')

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break

        self.assertAlmostEqual(m.devoted_acreage['CORN'].value, 80, 7)
        self.assertAlmostEqual(m.devoted_acreage['SUGAR_BEETS'].value, 250, 7)
        self.assertAlmostEqual(m.devoted_acreage['WHEAT'].value, 170, 7)

    @unittest.skipIf(not mpi4py_available, 'mpi4py is not available.')
    @unittest.skipIf(not numpy_available, 'numpy is not available.')
    @unittest.skipIf(not ipopt_available, 'ipopt is not available.')
    def test_grothey(self):
        def create_master():
            m = pyo.ConcreteModel()
            m.y = pyo.Var(bounds=(1, None))
            m.eta = pyo.Var(bounds=(-10, None))
            m.obj = pyo.Objective(expr=m.y ** 2 + m.eta)
            return m

        def create_subproblem(master):
            m = pyo.ConcreteModel()
            m.x1 = pyo.Var()
            m.x2 = pyo.Var()
            m.y = pyo.Var()
            m.obj = pyo.Objective(expr=-m.x2)
            m.c1 = pyo.Constraint(expr=(m.x1 - 1) ** 2 + m.x2 ** 2 <= pyo.log(m.y))
            m.c2 = pyo.Constraint(expr=(m.x1 + 1) ** 2 + m.x2 ** 2 <= pyo.log(m.y))

            complicating_vars_map = pyo.ComponentMap()
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
        opt = pyo.SolverFactory('ipopt')

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break
        self.assertAlmostEqual(m.y.value, 2.721381, 4)
        self.assertAlmostEqual(m.eta.value, -0.0337568, 4)

    @unittest.skipIf(not mpi4py_available, 'mpi4py is not available.')
    @unittest.skipIf(not numpy_available, 'numpy is not available.')
    @unittest.skipIf(not cplex_available, 'cplex is not available.')
    def test_four_scen_farmer(self):
        class FourScenFarmer(object):
            def __init__(self):
                self.crops = ['WHEAT', 'CORN', 'SUGAR_BEETS']
                self.total_acreage = 500
                self.PriceQuota = {'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0}
                self.SubQuotaSellingPrice = {'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0}
                self.SuperQuotaSellingPrice = {'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0}
                self.CattleFeedRequirement = {'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0}
                self.PurchasePrice = {'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0}
                self.PlantingCostPerAcre = {'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0}
                self.scenarios = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario', 'Scenario4']
                self.crop_yield = dict()
                self.crop_yield['BelowAverageScenario'] = {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0}
                self.crop_yield['AverageScenario'] = {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0}
                self.crop_yield['AboveAverageScenario'] = {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}
                self.crop_yield['Scenario4'] = {'WHEAT':2.0, 'CORN':3.0, 'SUGAR_BEETS':24.0}
                self.scenario_probabilities = dict()
                self.scenario_probabilities['BelowAverageScenario'] = 0.25
                self.scenario_probabilities['AverageScenario'] = 0.25
                self.scenario_probabilities['AboveAverageScenario'] = 0.25
                self.scenario_probabilities['Scenario4'] = 0.25

        def create_master(farmer):
            m = pyo.ConcreteModel()

            m.crops = pyo.Set(initialize=farmer.crops, ordered=True)
            m.scenarios = pyo.Set(initialize=farmer.scenarios, ordered=True)

            m.devoted_acreage = pyo.Var(m.crops, bounds=(0, farmer.total_acreage))
            m.eta = pyo.Var(m.scenarios)
            for s in m.scenarios:
                m.eta[s].setlb(-432000 * farmer.scenario_probabilities[s])

            m.total_acreage_con = pyo.Constraint(expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage)

            m.obj = pyo.Objective(
                expr=sum(farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop] for crop in m.crops) + sum(
                    m.eta.values()))
            return m

        def create_subproblem(master, farmer, scenario):
            m = pyo.ConcreteModel()

            m.crops = pyo.Set(initialize=farmer.crops, ordered=True)

            m.devoted_acreage = pyo.Var(m.crops)
            m.QuantitySubQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
            m.QuantitySuperQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
            m.QuantityPurchased = pyo.Var(m.crops, bounds=(0.0, None))

            def EnforceCattleFeedRequirement_rule(m, i):
                return (farmer.CattleFeedRequirement[i] <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) +
                        m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i])

            m.EnforceCattleFeedRequirement = pyo.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

            def LimitAmountSold_rule(m, i):
                return m.QuantitySubQuotaSold[i] + m.QuantitySuperQuotaSold[i] - (
                        farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) <= 0.0

            m.LimitAmountSold = pyo.Constraint(m.crops, rule=LimitAmountSold_rule)

            def EnforceQuotas_rule(m, i):
                return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])

            m.EnforceQuotas = pyo.Constraint(m.crops, rule=EnforceQuotas_rule)

            obj_expr = sum(farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops)
            obj_expr -= sum(farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops)
            obj_expr -= sum(farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops)
            m.obj = pyo.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)

            complicating_vars_map = pyo.ComponentMap()
            for crop in m.crops:
                complicating_vars_map[master.devoted_acreage[crop]] = m.devoted_acreage[crop]

            return m, complicating_vars_map

        farmer = FourScenFarmer()
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
                                     subproblem_solver='cplex_direct')
        opt = pyo.SolverFactory('cplex_direct')

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break

        self.assertAlmostEqual(m.devoted_acreage['CORN'].value ,100, 7)
        self.assertAlmostEqual(m.devoted_acreage['SUGAR_BEETS'].value, 250, 7)
        self.assertAlmostEqual(m.devoted_acreage['WHEAT'].value, 150, 7)


