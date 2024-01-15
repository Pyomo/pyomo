#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import sys
import time

from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo


"""
To run this example:

mpirun -np 3 python farmer.py
"""


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
        self.scenarios = [
            'BelowAverageScenario',
            'AverageScenario',
            'AboveAverageScenario',
        ]
        self.crop_yield = dict()
        self.crop_yield['BelowAverageScenario'] = {
            'WHEAT': 2.0,
            'CORN': 2.4,
            'SUGAR_BEETS': 16.0,
        }
        self.crop_yield['AverageScenario'] = {
            'WHEAT': 2.5,
            'CORN': 3.0,
            'SUGAR_BEETS': 20.0,
        }
        self.crop_yield['AboveAverageScenario'] = {
            'WHEAT': 3.0,
            'CORN': 3.6,
            'SUGAR_BEETS': 24.0,
        }
        self.scenario_probabilities = dict()
        self.scenario_probabilities['BelowAverageScenario'] = 0.3333
        self.scenario_probabilities['AverageScenario'] = 0.3334
        self.scenario_probabilities['AboveAverageScenario'] = 0.3333


def create_root(farmer):
    m = pyo.ConcreteModel()

    m.crops = pyo.Set(initialize=farmer.crops, ordered=True)
    m.scenarios = pyo.Set(initialize=farmer.scenarios, ordered=True)

    m.devoted_acreage = pyo.Var(m.crops, bounds=(0, farmer.total_acreage))
    m.eta = pyo.Var(m.scenarios)
    for s in m.scenarios:
        m.eta[s].setlb(-432000 * farmer.scenario_probabilities[s])

    m.total_acreage_con = pyo.Constraint(
        expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage
    )

    m.obj = pyo.Objective(
        expr=sum(
            farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop]
            for crop in m.crops
        )
        + sum(m.eta.values())
    )
    return m


def create_subproblem(root, farmer, scenario):
    m = pyo.ConcreteModel()

    m.crops = pyo.Set(initialize=farmer.crops, ordered=True)

    m.devoted_acreage = pyo.Var(m.crops)
    m.QuantitySubQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
    m.QuantitySuperQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
    m.QuantityPurchased = pyo.Var(m.crops, bounds=(0.0, None))

    def EnforceCattleFeedRequirement_rule(m, i):
        return (
            farmer.CattleFeedRequirement[i]
            <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i])
            + m.QuantityPurchased[i]
            - m.QuantitySubQuotaSold[i]
            - m.QuantitySuperQuotaSold[i]
        )

    m.EnforceCattleFeedRequirement = pyo.Constraint(
        m.crops, rule=EnforceCattleFeedRequirement_rule
    )

    def LimitAmountSold_rule(m, i):
        return (
            m.QuantitySubQuotaSold[i]
            + m.QuantitySuperQuotaSold[i]
            - (farmer.crop_yield[scenario][i] * m.devoted_acreage[i])
            <= 0.0
        )

    m.LimitAmountSold = pyo.Constraint(m.crops, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(m, i):
        return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])

    m.EnforceQuotas = pyo.Constraint(m.crops, rule=EnforceQuotas_rule)

    obj_expr = sum(
        farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops
    )
    obj_expr -= sum(
        farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop]
        for crop in m.crops
    )
    obj_expr -= sum(
        farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop]
        for crop in m.crops
    )
    m.obj = pyo.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)

    complicating_vars_map = pyo.ComponentMap()
    for crop in m.crops:
        complicating_vars_map[root.devoted_acreage[crop]] = m.devoted_acreage[crop]

    return m, complicating_vars_map


def main():
    rank = mpi4py.MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        sys.stdout = open(os.devnull, 'w')

    t0 = time.time()
    farmer = Farmer()
    m = create_root(farmer=farmer)
    root_vars = list(m.devoted_acreage.values())
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-8)
    for s in farmer.scenarios:
        subproblem_fn_kwargs = dict()
        subproblem_fn_kwargs['root'] = m
        subproblem_fn_kwargs['farmer'] = farmer
        subproblem_fn_kwargs['scenario'] = s
        m.benders.add_subproblem(
            subproblem_fn=create_subproblem,
            subproblem_fn_kwargs=subproblem_fn_kwargs,
            root_eta=m.eta[s],
            subproblem_solver='gurobi_persistent',
        )
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)

    print(
        '{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}'.format(
            '# Cuts', 'Corn', 'Sugar Beets', 'Wheat', 'Time'
        )
    )
    for i in range(30):
        res = opt.solve(tee=False, save_results=False)
        cuts_added = m.benders.generate_cut()
        for c in cuts_added:
            opt.add_constraint(c)
        print(
            '{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}'.format(
                len(cuts_added),
                m.devoted_acreage['CORN'].value,
                m.devoted_acreage['SUGAR_BEETS'].value,
                m.devoted_acreage['WHEAT'].value,
                time.time() - t0,
            )
        )
        if len(cuts_added) == 0:
            break


if __name__ == '__main__':
    main()
