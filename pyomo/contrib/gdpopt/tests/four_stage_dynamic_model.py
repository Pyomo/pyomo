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
from pyomo.core import (
    Var,
    Constraint,
    Objective,
    Set,
    minimize,
    exp,
    ConcreteModel,
    LogicalConstraint,
    exactly,
    lnot,
    lor,
    BooleanVar,
    land,
)
from pyomo.dae import Integral, DerivativeVar, ContinuousSet
from pyomo.gdp import Disjunct, Disjunction


def build_model(mode_transfer=False):
    model = ConcreteModel()

    # Set
    model.stage = Set(initialize=[1, 2, 3, 4])
    model.mode = Set(initialize=[1, 2, 3])

    model.t1 = ContinuousSet(bounds=(0, 1))
    model.t2 = ContinuousSet(bounds=(1, 2))
    model.t3 = ContinuousSet(bounds=(2, 3))
    model.t4 = ContinuousSet(bounds=(3, 4))

    # Variables
    model.x1 = Var(model.t1, bounds=(0, 10))
    model.x2 = Var(model.t2, bounds=(0, 10))
    model.x3 = Var(model.t3, bounds=(0, 10))
    model.x4 = Var(model.t4, bounds=(0, 10))
    model.u1 = Var(bounds=(-4, 4))
    model.u2 = Var(bounds=(-4, 4))
    model.u3 = Var(bounds=(-4, 4))
    model.u4 = Var(bounds=(-4, 4))

    # Dynamic model
    model.dxdt1 = DerivativeVar(model.x1, wrt=model.t1)
    model.dxdt2 = DerivativeVar(model.x2, wrt=model.t2)
    model.dxdt3 = DerivativeVar(model.x3, wrt=model.t3)
    model.dxdt4 = DerivativeVar(model.x4, wrt=model.t4)

    # logic constraint
    model.stage_mode = Disjunct(model.stage * model.mode)
    model.d = Disjunction(model.stage)

    # Stage 1

    def stage1_mode1_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt1[t] == -model.x1[t] * exp(model.x1[t] - 1) + model.u1

    model.stage_mode[1, 1].mode1_dynamic_constraint = Constraint(
        model.t1, rule=stage1_mode1_dynamic
    )

    def stage1_mode2_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt1[t] == (0.5 * model.x1[t] ** 3 + model.u1) / 20

    model.stage_mode[1, 2].mode2_dynamic_constraint = Constraint(
        model.t1, rule=stage1_mode2_dynamic
    )

    def stage1_mode3_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt1[t] == (model.x1[t] ** 2 + model.u1) / (t + 20)

    model.stage_mode[1, 3].mode3_dynamic_constraint = Constraint(
        model.t1, rule=stage1_mode3_dynamic
    )

    # Stage 2

    def stage2_mode1_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt2[t] == -model.x2[t] * exp(model.x2[t] - 1) + model.u2

    model.stage_mode[2, 1].mode1_dynamic_constraint = Constraint(
        model.t2, rule=stage2_mode1_dynamic
    )

    def stage2_mode2_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt2[t] == (0.5 * model.x2[t] ** 3 + model.u2) / 20

    model.stage_mode[2, 2].mode2_dynamic_constraint = Constraint(
        model.t2, rule=stage2_mode2_dynamic
    )

    def stage2_mode3_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt2[t] == (model.x2[t] ** 2 + model.u2) / (t + 20)

    model.stage_mode[2, 3].mode3_dynamic_constraint = Constraint(
        model.t2, rule=stage2_mode3_dynamic
    )

    # Stage 3

    def stage3_mode1_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt3[t] == -model.x3[t] * exp(model.x3[t] - 1) + model.u3

    model.stage_mode[3, 1].mode1_dynamic_constraint = Constraint(
        model.t3, rule=stage3_mode1_dynamic
    )

    def stage3_mode2_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt3[t] == (0.5 * model.x3[t] ** 3 + model.u3) / 20

    model.stage_mode[3, 2].mode2_dynamic_constraint = Constraint(
        model.t3, rule=stage3_mode2_dynamic
    )

    def stage3_mode3_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt3[t] == (model.x3[t] ** 2 + model.u3) / (t + 20)

    model.stage_mode[3, 3].mode3_dynamic_constraint = Constraint(
        model.t3, rule=stage3_mode3_dynamic
    )

    # Stage 4

    def stage4_mode1_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt4[t] == -model.x4[t] * exp(model.x4[t] - 1) + model.u4

    model.stage_mode[4, 1].mode1_dynamic_constraint = Constraint(
        model.t4, rule=stage4_mode1_dynamic
    )

    def stage4_mode2_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt4[t] == (0.5 * model.x4[t] ** 3 + model.u4) / 20

    model.stage_mode[4, 2].mode2_dynamic_constraint = Constraint(
        model.t4, rule=stage4_mode2_dynamic
    )

    def stage4_mode3_dynamic(disjunct, t):
        model = disjunct.model()
        return model.dxdt4[t] == (model.x4[t] ** 2 + model.u4) / (t + 20)

    model.stage_mode[4, 3].mode3_dynamic_constraint = Constraint(
        model.t4, rule=stage4_mode3_dynamic
    )

    model.d[1] = [
        model.stage_mode[1, 1],
        model.stage_mode[1, 2],
        model.stage_mode[1, 3],
    ]
    model.d[2] = [
        model.stage_mode[2, 1],
        model.stage_mode[2, 2],
        model.stage_mode[2, 3],
    ]
    model.d[3] = [
        model.stage_mode[3, 1],
        model.stage_mode[3, 2],
        model.stage_mode[3, 3],
    ]
    model.d[4] = [
        model.stage_mode[4, 1],
        model.stage_mode[4, 2],
        model.stage_mode[4, 3],
    ]

    if mode_transfer:
        model.lc1 = LogicalConstraint(
            expr=exactly(
                1,
                model.stage_mode[1, 1].indicator_var,
                model.stage_mode[1, 2].indicator_var,
                model.stage_mode[1, 3].indicator_var,
            )
        )
        model.lc2 = LogicalConstraint(
            expr=exactly(
                1,
                model.stage_mode[2, 1].indicator_var,
                model.stage_mode[2, 2].indicator_var,
                model.stage_mode[2, 3].indicator_var,
            )
        )
        model.lc3 = LogicalConstraint(
            expr=exactly(
                1,
                model.stage_mode[3, 1].indicator_var,
                model.stage_mode[3, 2].indicator_var,
                model.stage_mode[3, 3].indicator_var,
            )
        )
        model.lc4 = LogicalConstraint(
            expr=exactly(
                1,
                model.stage_mode[4, 1].indicator_var,
                model.stage_mode[4, 2].indicator_var,
                model.stage_mode[4, 3].indicator_var,
            )
        )
        model.transfer_stage1 = Set(initialize=[2, 3, 4])
        model.transfer_stage2 = Set(initialize=[2, 3, 4, 5])
        model.mode_stransfer_set = Set(initialize=[1, 2])
        model.mode_transfer = BooleanVar(
            model.transfer_stage2, model.mode_stransfer_set
        )
        model.mode_transfer_lc1 = LogicalConstraint(
            expr=exactly(
                1,
                model.mode_transfer[2, 1],
                model.mode_transfer[3, 1],
                model.mode_transfer[4, 1],
                model.mode_transfer[5, 1],
            )
        )
        model.mode_transfer_lc2 = LogicalConstraint(
            expr=exactly(
                1,
                model.mode_transfer[2, 2],
                model.mode_transfer[3, 2],
                model.mode_transfer[4, 2],
                model.mode_transfer[5, 2],
            )
        )

        def _mode_transfer_rule1(model, stage):
            return model.mode_transfer[stage, 1].equivalent_to(
                land(
                    model.stage_mode[stage - 1, 1].indicator_var,
                    model.stage_mode[stage, 2].indicator_var,
                )
            )

        model.mode_transfer2mode_choice_lc1 = LogicalConstraint(
            model.transfer_stage1, rule=_mode_transfer_rule1
        )

        def _mode_transfer_rule2(model, stage):
            return model.mode_transfer[stage, 2].equivalent_to(
                land(
                    model.stage_mode[stage - 1, 2].indicator_var,
                    model.stage_mode[stage, 3].indicator_var,
                )
            )

        model.mode_transfer2mode_choice_lc2 = LogicalConstraint(
            model.transfer_stage1, rule=_mode_transfer_rule2
        )

        def _mode_transfer_rule3(model, stage):
            return model.mode_transfer[stage, 2].implies(
                lor(
                    model.mode_transfer[stage1, 1]
                    for stage1 in model.transfer_stage1
                    if stage1 < stage
                )
            )

        model.mode_transfer2mode_choice_lc3 = LogicalConstraint(
            model.transfer_stage1, rule=_mode_transfer_rule3
        )

        def _mode_transfer_rule4(model):
            return model.mode_transfer[5, 1].implies(
                lnot(
                    lor(
                        model.stage_mode[stage1, 2].indicator_var
                        for stage1 in model.stage
                    )
                )
            )

        model.mode_transfer2mode_choice_lc4 = LogicalConstraint(
            rule=_mode_transfer_rule4
        )

        def _mode_transfer_rule5(model):
            return model.mode_transfer[5, 2].implies(
                lnot(
                    lor(
                        model.stage_mode[stage1, 3].indicator_var
                        for stage1 in model.stage
                    )
                )
            )

        model.mode_transfer2mode_choice_lc5 = LogicalConstraint(
            rule=_mode_transfer_rule5
        )

    # Sequence constraint
    def _sequence_rule1(model, stage):
        if stage == 1:
            return Constraint.Skip
        else:
            return model.stage_mode[stage, 2].indicator_var.implies(
                lor(
                    model.stage_mode[stage2, 1].indicator_var
                    for stage2 in model.stage
                    if stage2 < stage
                )
            )

    model.seq1 = LogicalConstraint(model.stage, rule=_sequence_rule1)
    model.stage_mode[1, 2].indicator_var.fix(False)

    def _sequence_rule2(model, stage):
        if stage == 4:
            return Constraint.Skip
        else:
            return model.stage_mode[stage, 2].indicator_var.implies(
                lnot(
                    lor(
                        model.stage_mode[stage2, 1].indicator_var
                        for stage2 in model.stage
                        if stage2 > stage
                    )
                )
            )

    model.seq2 = LogicalConstraint(model.stage, rule=_sequence_rule2)

    def _sequence_rule3(model, stage):
        if stage <= 1:
            return Constraint.Skip
        else:
            return model.stage_mode[stage, 3].indicator_var.implies(
                lor(
                    model.stage_mode[stage2, 2].indicator_var
                    for stage2 in model.stage
                    if stage2 < stage
                )
            )

    model.seq3 = LogicalConstraint(model.stage, rule=_sequence_rule3)
    model.stage_mode[1, 3].indicator_var.fix(False)
    model.stage_mode[2, 3].indicator_var.fix(False)

    def _sequence_rule4(model, stage):
        if stage == 4:
            return Constraint.Skip
        else:
            return model.stage_mode[stage, 3].indicator_var.implies(
                lnot(
                    lor(
                        model.stage_mode[stage2, 2].indicator_var
                        for stage2 in model.stage
                        if stage2 > stage
                    )
                )
            )

    model.seq4 = LogicalConstraint(model.stage, rule=_sequence_rule4)

    model.c1 = Constraint(expr=model.x1[0] == 1)
    model.c2 = Constraint(expr=model.x1[1] == model.x2[1])
    model.c3 = Constraint(expr=model.x2[2] == model.x3[2])
    model.c4 = Constraint(expr=model.x3[3] == model.x4[3])

    # Objective function
    model.intx1 = Integral(
        model.t1, wrt=model.t1, rule=lambda model, t: model.x1[t] ** 2
    )
    model.intx2 = Integral(
        model.t2, wrt=model.t2, rule=lambda model, t: model.x2[t] ** 2
    )
    model.intx3 = Integral(
        model.t3, wrt=model.t3, rule=lambda model, t: model.x3[t] ** 2
    )
    model.intx4 = Integral(
        model.t4, wrt=model.t4, rule=lambda model, t: model.x4[t] ** 2
    )

    model.obj = Objective(
        expr=-(model.intx1 + model.intx2 + model.intx3 + model.intx4), sense=minimize
    )
    return model
