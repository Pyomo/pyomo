#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
import networkx


def create_model(demand_factor=1.0):

    model = pyo.ConcreteModel()

    # sets
    model.TIME = dae.ContinuousSet(bounds=(0.0, 24.0))
    model.DIS = dae.ContinuousSet(bounds=(0.0, 1.0))
    model.S = pyo.Param(initialize=1)
    model.SCEN = pyo.RangeSet(1, model.S)

    # links
    model.LINK = pyo.Set(initialize=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12'])

    def rule_startloc(m, l):
        ll = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12']
        ls = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12']
        start_locations = dict(zip(ll, ls))
        return start_locations[l]

    model.lstartloc = pyo.Param(model.LINK, initialize=rule_startloc, within=pyo.Any)

    def rule_endloc(m, l):
        ll = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12']
        ls = ['n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13']
        end_locations = dict(zip(ll, ls))
        return end_locations[l]
    model.lendloc = pyo.Param(model.LINK, initialize=rule_endloc, within=pyo.Any)

    model.ldiam = pyo.Param(model.LINK, initialize=920.0, mutable=True)

    def rule_llength(m, l):
        if l == 'l1' or l == 'l12':
            return 300.0
        return 100.0
    model.llength = pyo.Param(model.LINK, initialize=rule_llength, mutable=True)

    def rule_ltype(m, l):
        if l == 'l1' or l == 'l12':
            return 'p'
        return 'a'
    model.ltype = pyo.Param(model.LINK, initialize=rule_ltype, within=pyo.Any)

    def link_a_init_rule(m):
        return (l for l in m.LINK if m.ltype[l] == "a")
    model.LINK_A = pyo.Set(initialize=link_a_init_rule)

    def link_p_init_rule(m):
        return (l for l in m.LINK if m.ltype[l] == "p")
    model.LINK_P = pyo.Set(initialize=link_p_init_rule)

    # nodes
    model.NODE = pyo.Set(initialize=['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13'])

    def rule_pmin(m, n):
        if n == 'n1':
            return 57.0
        elif n == 'n13':
            return 39.0
        else:
            return 34.0

    model.pmin = pyo.Param(model.NODE, initialize=rule_pmin, mutable=True)

    def rule_pmax(m, n):
        if n == 'n13':
            return 41.0
        return 70.0
    model.pmax = pyo.Param(model.NODE, initialize=rule_pmax, mutable=True)

    # supply
    model.SUP = pyo.Set(initialize=[1])
    model.sloc = pyo.Param(model.SUP, initialize='n1', within=pyo.Any)
    model.smin = pyo.Param(model.SUP, within=pyo.NonNegativeReals, initialize=0.000, mutable=True)
    model.smax = pyo.Param(model.SUP, within=pyo.NonNegativeReals, initialize=30, mutable=True)
    model.scost = pyo.Param(model.SUP, within=pyo.NonNegativeReals)

    # demand
    model.DEM = pyo.Set(initialize=[1])
    model.dloc = pyo.Param(model.DEM, initialize='n13', within=pyo.Any)
    model.d = pyo.Param(model.DEM, within=pyo.PositiveReals, initialize=10, mutable=True)

    # physical data

    model.TDEC = pyo.Param(initialize=9.5)

    model.eps = pyo.Param(initialize=0.025, within=pyo.PositiveReals)
    model.z = pyo.Param(initialize=0.80, within=pyo.PositiveReals)
    model.rhon = pyo.Param(initialize=0.72, within=pyo.PositiveReals)
    model.R = pyo.Param(initialize=8314.0, within=pyo.PositiveReals)
    model.M = pyo.Param(initialize=18.0, within=pyo.PositiveReals)
    model.pi = pyo.Param(initialize=3.14, within=pyo.PositiveReals)
    model.nu2 = pyo.Param(within=pyo.PositiveReals,mutable=True)
    model.lam = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
    model.A = pyo.Param(model.LINK, within=pyo.NonNegativeReals, mutable=True)
    model.Tgas = pyo.Param(initialize=293.15, within=pyo.PositiveReals)
    model.Cp = pyo.Param(initialize=2.34, within=pyo.PositiveReals)
    model.Cv = pyo.Param(initialize=1.85, within=pyo.PositiveReals)
    model.gam = pyo.Param(initialize=model.Cp/model.Cv, within=pyo.PositiveReals)
    model.om = pyo.Param(initialize=(model.gam-1.0)/model.gam, within=pyo.PositiveReals)

    # scaling and constants
    model.ffac = pyo.Param(within=pyo.PositiveReals, initialize=(1.0e+6*model.rhon)/(24.0*3600.0))
    model.ffac2 = pyo.Param(within=pyo.PositiveReals, initialize=3600.0/(1.0e+4 * model.rhon))
    model.pfac = pyo.Param(within=pyo.PositiveReals, initialize=1.0e+5)
    model.pfac2 = pyo.Param(within=pyo.PositiveReals, initialize=1.0e-5)
    model.dfac = pyo.Param(within=pyo.PositiveReals, initialize=1.0e-3)
    model.lfac = pyo.Param(within=pyo.PositiveReals, initialize=1.0e+3)

    model.c1 = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
    model.c2 = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
    model.c3 = pyo.Param(model.LINK, within=pyo.PositiveReals, mutable=True)
    model.c4 = pyo.Param(within=pyo.PositiveReals, mutable=True)

    # cost factors
    model.ce = pyo.Param(initialize=0.1, within=pyo.NonNegativeReals)
    model.cd = pyo.Param(initialize=1.0e+6, within=pyo.NonNegativeReals)
    model.cT = pyo.Param(initialize=1.0e+6, within=pyo.NonNegativeReals)
    model.cs = pyo.Param(initialize=0.0, within=pyo.NonNegativeReals)

    # define stochastic info
    model.rand_d = pyo.Param(model.SCEN, model.DEM, within=pyo.NonNegativeReals, mutable=True)

    # convert units for input data
    def rescale_rule(m):

        for i in m.LINK:
            m.ldiam[i] = m.ldiam[i]*m.dfac
            m.llength[i] = m.llength[i]*m.lfac
            # m.dx[i] = m.llength[i]/float(m.DIS.last())

        for i in m.SUP:
            m.smin[i] = m.smin[i]*m.ffac*m.ffac2   # from scmx106/day to kg/s and then to scmx10-4/hr
            m.smax[i] = m.smax[i]*m.ffac*m.ffac2   # from scmx106/day to kg/s and then to scmx10-4/hr

        for i in m.DEM:
            m.d[i] = m.d[i]*m.ffac*m.ffac2

        for i in m.NODE:
            m.pmin[i] = m.pmin[i]*m.pfac*m.pfac2   # from bar to Pascals and then to bar
            m.pmax[i] = m.pmax[i]*m.pfac*m.pfac2   # from bar to Pascals and then to bar
    rescale_rule(model)

    def compute_constants(m):
        for i in m.LINK:
            m.lam[i] = (2.0*pyo.log10(3.7*m.ldiam[i]/(m.eps*m.dfac)))**(-2.0)
            m.A[i] = (1.0/4.0)*m.pi*m.ldiam[i]*m.ldiam[i]
            m.nu2 = m.gam*m.z*m.R*m.Tgas/m.M
            m.c1[i] = (m.pfac2/m.ffac2)*(m.nu2/m.A[i])
            m.c2[i] = m.A[i]*(m.ffac2/m.pfac2)
            m.c3[i] = m.A[i]*(m.pfac2/m.ffac2)*(8.0*m.lam[i]*m.nu2)/(m.pi*m.pi*(m.ldiam[i]**5.0))
            m.c4 = (1/m.ffac2)*(m.Cp*m.Tgas)
    compute_constants(model)

    # set stochastic demands
    def compute_demands_rule(m):
        for k in m.SCEN:
            for j in m.DEM:
                m.rand_d[k, j] = demand_factor*m.d[j]

    compute_demands_rule(model)

    def stochd_init(m, k, j, t):
        # What it should be to match description in paper
        # if t < m.TDEC:
        #     return m.d[j]
        # if t >= m.TDEC and t < m.TDEC+5:
        #     return m.rand_d[k,j]
        # if t >= m.TDEC+5:
        #     return m.d[j]
        if t < m.TDEC+1:
            return m.d[j]
        if t >= m.TDEC+1 and t < m.TDEC+1+4.5:
            return m.rand_d[k, j]
        if t >= m.TDEC+1+4.5:
            return m.d[j]

    model.stochd = pyo.Param(model.SCEN, model.DEM, model.TIME, within=pyo.PositiveReals, mutable=True, default=stochd_init)

    # define temporal variables
    def p_bounds_rule(m, k, j, t):
        return pyo.value(m.pmin[j]), pyo.value(m.pmax[j])
    model.p = pyo.Var(model.SCEN, model.NODE, model.TIME, bounds=p_bounds_rule, initialize=50.0)


    model.dp = pyo.Var(model.SCEN, model.LINK_A, model.TIME, bounds=(0.0, 100.0), initialize=10.0)
    model.fin = pyo.Var(model.SCEN, model.LINK, model.TIME, bounds=(1.0, 500.0), initialize=100.0)
    model.fout = pyo.Var(model.SCEN, model.LINK, model.TIME, bounds=(1.0, 500.0), initialize=100.0)

    def s_bounds_rule(m, k, j, t):
        return 0.01, pyo.value(m.smax[j])
    model.s = pyo.Var(model.SCEN, model.SUP, model.TIME, bounds=s_bounds_rule, initialize=10.0)
    model.dem = pyo.Var(model.SCEN, model.DEM, model.TIME, initialize=100.0)
    model.pow = pyo.Var(model.SCEN, model.LINK_A, model.TIME, bounds=(0.0, 3000.0), initialize=1000.0)
    model.slack = pyo.Var(model.SCEN, model.LINK, model.TIME, model.DIS, bounds=(0.0, None), initialize=10.0)

    # define spatio-temporal variables
    # average 55.7278214666423
    model.px = pyo.Var(model.SCEN, model.LINK, model.TIME, model.DIS, bounds=(10.0, 100.0), initialize=50.0)
    # average 43.19700578593625
    model.fx = pyo.Var(model.SCEN, model.LINK, model.TIME, model.DIS, bounds=(1.0, 100.0), initialize=100.0)

    # define derivatives
    model.dpxdt = dae.DerivativeVar(model.px, wrt=model.TIME, initialize=0)
    model.dpxdx = dae.DerivativeVar(model.px, wrt=model.DIS, initialize=0)
    model.dfxdt = dae.DerivativeVar(model.fx, wrt=model.TIME, initialize=0)
    model.dfxdx = dae.DerivativeVar(model.fx, wrt=model.DIS, initialize=0)


    # ----------- MODEL --------------

    # compressor equations
    def powereq_rule(m, j, i, t):
        return m.pow[j, i, t] == m.c4 * m.fin[j, i, t] * (((m.p[j, m.lstartloc[i], t]+m.dp[j, i, t])/m.p[j, m.lstartloc[i], t])**m.om - 1.0)
    model.powereq = pyo.Constraint(model.SCEN, model.LINK_A, model.TIME, rule=powereq_rule)

    # cvar model
    model.cvar_lambda = pyo.Param(initialize=0.0)
    model.nu = pyo.Var(initialize=100.0)
    model.phi = pyo.Var(model.SCEN, bounds=(0.0, None), initialize=100.0)

    def cvarcost_rule(m):
        return (1.0/m.S) * sum((m.phi[k]/(1.0-0.95) + m.nu) for k in m.SCEN)
    model.cvarcost = pyo.Expression(rule=cvarcost_rule)

    # node balances
    def nodeeq_rule(m, k, i, t):
        return sum(m.fout[k, j, t] for j in m.LINK if m.lendloc[j] == i) +  \
               sum(m.s[k, j, t] for j in m.SUP if m.sloc[j] == i) -         \
               sum(m.fin[k, j, t] for j in m.LINK if m.lstartloc[j] == i) - \
               sum(m.dem[k, j, t] for j in m.DEM if m.dloc[j] == i) == 0.0
    model.nodeeq = pyo.Constraint(model.SCEN, model.NODE, model.TIME, rule=nodeeq_rule)

    # boundary conditions flow
    def flow_start_rule(m, j, i, t):
        return m.fx[j, i, t, m.DIS.first()] == m.fin[j, i, t]
    model.flow_start = pyo.Constraint(model.SCEN, model.LINK, model.TIME, rule=flow_start_rule)

    def flow_end_rule(m, j, i, t):
        return m.fx[j, i, t, m.DIS.last()] == m.fout[j, i, t]
    model.flow_end = pyo.Constraint(model.SCEN, model.LINK, model.TIME, rule=flow_end_rule)

    # First PDE for gas network model
    def flow_rule(m, j, i, t, k):
        if t == m.TIME.first() or k == m.DIS.last():
            return pyo.Constraint.Skip # Do not apply pde at initial time or final location
        return m.dpxdt[j, i, t, k]/3600.0 + m.c1[i]/m.llength[i] * m.dfxdx[j, i, t, k] == 0
    model.flow = pyo.Constraint(model.SCEN, model.LINK, model.TIME, model.DIS, rule=flow_rule)

    # Second PDE for gas network model
    def press_rule(m, j, i, t, k):
        if t == m.TIME.first() or k == m.DIS.last():
            return pyo.Constraint.Skip # Do not apply pde at initial time or final location
        return m.dfxdt[j, i, t, k]/3600 == -m.c2[i]/m.llength[i]*m.dpxdx[j, i, t, k] - m.slack[j, i, t, k]
    model.press = pyo.Constraint(model.SCEN, model.LINK, model.TIME, model.DIS, rule=press_rule)

    def slackeq_rule(m, j, i, t, k):
        if t == m.TIME.last():
            return pyo.Constraint.Skip
        return m.slack[j, i, t, k] * m.px[j, i, t, k] == m.c3[i] * m.fx[j, i, t, k] * m.fx[j, i, t, k]
    model.slackeq = pyo.Constraint(model.SCEN, model.LINK, model.TIME, model.DIS, rule=slackeq_rule)

    # boundary conditions pressure, passive links
    def presspas_start_rule(m, j, i, t):
        return m.px[j, i, t, m.DIS.first()] == m.p[j, m.lstartloc[i], t]
    model.presspas_start = pyo.Constraint(model.SCEN, model.LINK_P, model.TIME, rule=presspas_start_rule)

    def presspas_end_rule(m, j, i, t):
        return m.px[j, i, t, m.DIS.last()] == m.p[j, m.lendloc[i], t]
    model.presspas_end = pyo.Constraint(model.SCEN, model.LINK_P, model.TIME, rule=presspas_end_rule)

    # boundary conditions pressure, active links
    def pressact_start_rule(m, j, i, t):
        return m.px[j, i, t, m.DIS.first()] == m.p[j, m.lstartloc[i], t] + m.dp[j, i, t]
    model.pressact_start = pyo.Constraint(model.SCEN, model.LINK_A, model.TIME, rule=pressact_start_rule)

    def pressact_end_rule(m, j, i, t):
        return m.px[j, i, t, m.DIS.last()] == m.p[j, m.lendloc[i], t]
    model.pressact_end = pyo.Constraint(model.SCEN, model.LINK_A, model.TIME, rule=pressact_end_rule)

    # fix pressure at supply nodes
    def suppres_rule(m, k, j, t):
        return m.p[k, m.sloc[j], t] == m.pmin[m.sloc[j]]
    model.suppres = pyo.Constraint(model.SCEN, model.SUP, model.TIME, rule=suppres_rule)

    # discharge pressure for compressors
    def dispress_rule(m, j, i, t):
        return m.p[j, m.lstartloc[i], t] + m.dp[j, i, t] <= m.pmax[m.lstartloc[i]]
    model.dispress = pyo.Constraint(model.SCEN, model.LINK_A, model.TIME, rule=dispress_rule)

    # ss constraints
    def flow_ss_rule(m, j, i, k):
        if k == m.DIS.last():
            return pyo.Constraint.Skip
        return m.dfxdx[j, i, m.TIME.first(), k] == 0.0
    model.flow_ss = pyo.Constraint(model.SCEN, model.LINK, model.DIS, rule=flow_ss_rule)

    def pres_ss_rule(m, j, i, k):
        if k == m.DIS.last():
            return pyo.Constraint.Skip
        return 0.0 == - m.c2[i]/m.llength[i] * m.dpxdx[j, i, m.TIME.first(), k] - m.slack[j, i, m.TIME.first(), k];
    model.pres_ss = pyo.Constraint(model.SCEN, model.LINK, model.DIS, rule=pres_ss_rule)

    # non-anticipativity constraints
    def nonantdq_rule(m, j, i, t):
        if j == 1:
            return pyo.Constraint.Skip
        if t >= m.TDEC+1:
            return pyo.Constraint.Skip
        return m.dp[j, i, t] == m.dp[1, i, t]

    model.nonantdq = pyo.Constraint(model.SCEN, model.LINK_A, model.TIME, rule=nonantdq_rule)

    def nonantde_rule(m, j, i, t):
        if j == 1:
            return pyo.Constraint.Skip
        if t >= m.TDEC+1:
            return pyo.Constraint.Skip
        return m.dem[j, i, t] == m.dem[1, i, t]

    model.nonantde = pyo.Constraint(model.SCEN, model.DEM, model.TIME, rule=nonantde_rule)

    # discretize model
    discretizer = pyo.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(model, nfe=1, wrt=model.DIS, scheme='FORWARD')

    discretizer2 = pyo.TransformationFactory('dae.collocation')
    #discretizer2.apply_to(model, nfe=47, ncp=1, wrt=model.TIME, scheme='LAGRANGE-RADAU')

    # discretizer.apply_to(model, nfe=48, wrt=model.TIME, scheme='BACKWARD')

    # What it should be to match description in paper
    discretizer.apply_to(model, nfe=48, wrt=model.TIME, scheme='BACKWARD')

    TimeStep = model.TIME[2] - model.TIME[1]

    def supcost_rule(m, k):
        return sum(m.cs * m.s[k, j, t] * TimeStep for j in m.SUP for t in m.TIME.get_finite_elements())

    model.supcost = pyo.Expression(model.SCEN, rule=supcost_rule)

    def boostcost_rule(m, k):
        return sum(m.ce * m.pow[k, j, t] * TimeStep for j in m.LINK_A for t in m.TIME.get_finite_elements())

    model.boostcost = pyo.Expression(model.SCEN, rule=boostcost_rule)

    def trackcost_rule(m, k):
        return sum(
            m.cd * (m.dem[k, j, t] - m.stochd[k, j, t]) ** 2.0 for j in m.DEM for t in m.TIME.get_finite_elements())

    model.trackcost = pyo.Expression(model.SCEN, rule=trackcost_rule)

    def sspcost_rule(m, k):
        return sum(
            m.cT * (m.px[k, i, m.TIME.last(), j] - m.px[k, i, m.TIME.first(), j]) ** 2.0 for i in m.LINK for j in m.DIS)

    model.sspcost = pyo.Expression(model.SCEN, rule=sspcost_rule)

    def ssfcost_rule(m, k):
        return sum(
            m.cT * (m.fx[k, i, m.TIME.last(), j] - m.fx[k, i, m.TIME.first(), j]) ** 2.0 for i in m.LINK for j in m.DIS)

    model.ssfcost = pyo.Expression(model.SCEN, rule=ssfcost_rule)

    def cost_rule(m, k):
        return 1e-6 * (m.supcost[k] + m.boostcost[k] + m.trackcost[k] + m.sspcost[k] + m.ssfcost[k])

    model.cost = pyo.Expression(model.SCEN, rule=cost_rule)

    def mcost_rule(m):
        return sum(m.cost[k] for k in m.SCEN)

    model.mcost = pyo.Expression(rule=mcost_rule)

    model.FirstStageCost = pyo.Expression(expr=0.0)
    model.SecondStageCost = pyo.Expression(rule=mcost_rule)

    model.obj = pyo.Objective(expr=model.FirstStageCost + model.SecondStageCost)

    return model

"""
instance = create_model(1.0)

solver = pyo.SolverFactory("ipopt")
solver.solve(instance, tee=True)

import sys
sys.exit()
"""

# Define the scenario tree with networkx
nx_scenario_tree = networkx.DiGraph()
# first stage
nx_scenario_tree.add_node("R",
                          cost="FirstStageCost",
                          variables=["dp", "dem"])
# second stage
demand_factors = np.random.uniform(0.8, 2.5, 5)
n_scenarios = len(demand_factors)
for i, df in enumerate(demand_factors):
    nx_scenario_tree.add_node("s{}".format(i),
                              cost="SecondStageCost")
    nx_scenario_tree.add_edge("R", "s{}".format(i), weight=1/n_scenarios)

# Creates an instance for each scenario
def pysp_instance_creation_callback(scenario_name, node_names):
    sid = scenario_name.strip("s")
    df = demand_factors[int(sid)]
    model = create_model(df)
    return model

