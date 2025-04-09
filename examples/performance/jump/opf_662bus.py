#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo


class Bus:
    pass


class Branch:
    pass


bus = []
busmap = {}
busfile = open("IEEE662.bus", "r")
for i, line in enumerate(busfile):
    sp = line.split()
    b = Bus()
    busmap[sp[0]] = i
    b.bustype = int(sp[1])
    b.name = sp[2]
    b.voltage0 = float(sp[3])
    b.angle0 = float(sp[4])
    b.p_gen = float(sp[5])
    b.q_gen = float(sp[6])
    b.q_min = float(sp[7])
    b.q_max = float(sp[8])
    b.p_load = float(sp[9])
    b.q_load = float(sp[10])
    b.g_shunt = float(sp[11])
    b.b_shunt0 = float(sp[12])
    b.b_shunt_min = float(sp[13])
    b.b_shunt_max = float(sp[14])
    b.b_dispatch = float(sp[15])
    b.area = float(sp[16])
    # rescale
    b.p_gen /= 100
    b.q_gen /= 100
    b.q_min /= 100
    b.q_max /= 100
    b.p_load /= 100
    b.q_load /= 100
    bus.append(b)

branchfile = open("IEEE662.branch", "r")
branch = []
for i, line in enumerate(branchfile):
    sp = line.split()
    b = Branch()
    b.frm = busmap[sp[1]]
    b.to = busmap[sp[2]]
    b.branchtype = int(sp[3])
    b.r = float(sp[4])
    b.x = float(sp[5])
    b.c = float(sp[6])
    b.tap0 = float(sp[7])
    b.tap_min0 = float(sp[8])
    b.tap_max0 = float(sp[9])
    b.def0 = float(sp[10])
    b.def_min = float(sp[11])
    b.def_max = float(sp[12])
    b.g = b.r / (b.r**2 + b.x**2)
    b.b = -b.x / (b.r**2 + b.x**2)
    b.def_min *= 3.14159 / 180
    b.def_max *= 3.14159 / 180
    b.def0 *= -3.14159 / 180
    branch.append(b)


bus_voltage_min = {0: 0.85, 1: 0.85, 2: 0.92, 3: 0.99}
bus_voltage_max = {0: 1.15, 1: 1.15, 2: 1.08, 3: 1.01}
branch_tap_min = 0.85
branch_tap_max = 1.15

p_gen_upper = 1.10
p_gen_lower = 0.90

nbus = len(bus)
nbranch = len(branch)

in_lines = [[] for i in range(nbus)]
out_lines = [[] for i in range(nbus)]
for i in range(nbranch):
    b = branch[i]
    out_lines[b.frm].append(i)
    in_lines[b.to].append(i)
    assert b.to >= 0 and b.to < nbus


model = pyo.ConcreteModel()

model.bus_voltage = pyo.Var(
    range(nbus),
    bounds=lambda model, i: (
        bus_voltage_min[bus[i].bustype],
        bus_voltage_max[bus[i].bustype],
    ),
    initialize=1,
)
model.bus_b_shunt = pyo.Var(
    range(nbus),
    bounds=lambda model, i: (bus[i].b_shunt_min, bus[i].b_shunt_max),
    initialize=lambda model, i: bus[i].b_shunt0,
)
model.bus_angle = pyo.Var(range(nbus), initialize=0)

model.branch_tap = pyo.Var(
    range(nbranch), bounds=(branch_tap_min, branch_tap_max), initialize=1
)
model.branch_def = pyo.Var(
    range(nbranch),
    bounds=lambda model, i: (branch[i].def_min, branch[i].def_max),
    initialize=lambda model, i: branch[i].def0,
)


def Gself(k):
    return (
        bus[k].g_shunt
        + sum(branch[i].g * model.branch_tap[i] ** 2 for i in out_lines[k])
        + sum(branch[i].g for i in in_lines[k])
    )


def Gout(i):
    return (
        -branch[i].g * pyo.cos(model.branch_def[i])
        + branch[i].b * pyo.sin(model.branch_def[i])
    ) * model.branch_tap[i]


def Gin(i):
    return (
        -branch[i].g * pyo.cos(model.branch_def[i])
        - branch[i].b * pyo.sin(model.branch_def[i])
    ) * model.branch_tap[i]


def Bself(k):
    return (
        model.bus_b_shunt[k]
        + sum(
            branch[i].b * model.branch_tap[i] ** 2 + branch[i].c / 2
            for i in out_lines[k]
        )
        + sum(branch[i].b + branch[i].c / 2 for i in in_lines[k])
    )


def Bin(i):
    return (
        branch[i].g * pyo.sin(model.branch_def[i])
        - branch[i].b * pyo.cos(model.branch_def[i])
    ) * model.branch_tap[i]


def Bout(i):
    return (
        -branch[i].g * pyo.sin(model.branch_def[i])
        - branch[i].b * pyo.cos(model.branch_def[i])
    ) * model.branch_tap[i]


model.obj = pyo.Objective(
    expr=sum(
        (
            bus[k].p_load
            + sum(
                model.bus_voltage[k]
                * model.bus_voltage[branch[i].frm]
                * (
                    Gin(i)
                    * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].frm])
                    + Bin(i)
                    * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].frm])
                )
                for i in in_lines[k]
            )
            + sum(
                model.bus_voltage[k]
                * model.bus_voltage[branch[i].to]
                * (
                    Gout(i)
                    * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].to])
                    + Bout(i)
                    * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].to])
                )
                for i in out_lines[k]
            )
            + model.bus_voltage[k] ** 2 * Gself(k)
        )
        ** 2
        for k in range(nbus)
        if bus[k].bustype == 2 or bus[k].bustype == 3
    )
)


def p_load_rule(model, k):
    if bus[k].bustype != 0:
        return pyo.Constraint.Skip

    return (
        bus[k].p_gen
        - bus[k].p_load
        - sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].frm]
            * (
                Gin(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].frm])
                + Bin(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].frm])
            )
            for i in in_lines[k]
        )
        - sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].to]
            * (
                Gout(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].to])
                + Bout(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].to])
            )
            for i in out_lines[k]
        )
        - model.bus_voltage[k] ** 2 * Gself(k)
        == 0
    )


model.p_load_constr = pyo.Constraint(range(nbus), rule=p_load_rule)


def q_load_rule(model, k):
    if bus[k].bustype != 0:
        return pyo.Constraint.Skip

    return (
        bus[k].q_gen
        - bus[k].q_load
        - sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].frm]
            * (
                Gin(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].frm])
                - Bin(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].frm])
            )
            for i in in_lines[k]
        )
        - sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].to]
            * (
                Gout(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].to])
                - Bout(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].to])
            )
            for i in out_lines[k]
        )
        + model.bus_voltage[k] ** 2 * Bself(k)
        == 0
    )


model.q_load_constr = pyo.Constraint(range(nbus), rule=q_load_rule)


def q_inj_rule(model, k):
    if not (bus[k].bustype == 2 or bus[k].bustype == 3):
        return pyo.Constraint.Skip

    return (
        bus[k].q_min,
        bus[k].q_load
        + sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].frm]
            * (
                Gin(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].frm])
                - Bin(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].frm])
            )
            for i in in_lines[k]
        )
        + sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].to]
            * (
                Gout(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].to])
                - Bout(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].to])
            )
            for i in out_lines[k]
        )
        - model.bus_voltage[k] ** 2 * Bself(k),
        bus[k].q_max,
    )


model.q_inj_rule = pyo.Constraint(range(nbus), rule=q_inj_rule)


def p_inj_rule(model, k):
    if not (bus[k].bustype == 2 or bus[k].bustype == 3):
        return pyo.Constraint.Skip

    return (
        0,
        bus[k].p_load
        + sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].frm]
            * (
                Gin(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].frm])
                + Bin(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].frm])
            )
            for i in in_lines[k]
        )
        + sum(
            model.bus_voltage[k]
            * model.bus_voltage[branch[i].to]
            * (
                Gout(i) * pyo.cos(model.bus_angle[k] - model.bus_angle[branch[i].to])
                + Bout(i) * pyo.sin(model.bus_angle[k] - model.bus_angle[branch[i].to])
            )
            for i in out_lines[k]
        )
        + model.bus_voltage[k] ** 2 * Gself(k),
        p_gen_upper * bus[k].p_gen,
    )


model.p_inj_rule = pyo.Constraint(range(nbus), rule=p_inj_rule)


for i in range(nbus):
    if bus[i].bustype == 3:
        model.bus_angle[i].fixed = True
    if bus[i].b_dispatch == 0:
        model.bus_b_shunt[i].fixed = True

for i in range(nbranch):
    if branch[i].branchtype == 3 or branch[i].branchtype == 0:
        model.branch_tap[i].fixed = True
    if branch[i].branchtype != 4:
        model.branch_def[i].fixed = True
