#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import ConcreteModel, Var, Param, Expression, Objective, Constraint, Piecewise, sum_product, inequality
from pyomo.pysp.annotations import (StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation)

d = {}
d['Scenario1'] = 0
d['Scenario2'] = 1
d['Scenario3'] = 2
def create_instance(scenario_name):
    cnt = d[scenario_name]

    model = ConcreteModel()
    # first stage
    model.x = Var(bounds=(0,10))
    # first stage derived
    model.y = Expression(expr=model.x + 1)
    model.fx = Var()
    # second stage
    model.z = Var(bounds=(-10, 10))
    # second stage derived
    model.q = Expression(expr=model.z**2)
    model.fz = Var()
    model.r = Var()
    # stage costs
    model.StageCost = Expression([1,2])
    model.StageCost.add(1, model.fx)
    model.StageCost.add(2, -model.fz + model.r - cnt)
    model.o = Objective(expr=sum_product(model.StageCost))

    model.ZERO = Param(initialize=0, mutable=True)
    if cnt == 0:
        cnt = model.ZERO

    model.c_first_stage = Constraint(expr= model.x >= 0)

    # test our handling of intermediate variables that
    # are created by Piecewise but can not necessarily
    # be declared on the scenario tree
    model.p_first_stage = Piecewise(model.fx, model.x,
                                    pw_pts=[0.,2.,5.,7.,10.],
                                    pw_constr_type='EQ',
                                    pw_repn='INC',
                                    f_rule=[10.,10.,9.,10.,10.],
                                    force_pw=True)

    model.c_second_stage = Constraint(expr= model.x + model.r * cnt >= -100)
    model.r_second_stage = Constraint(expr= inequality(-cnt, model.r, 0))
    # exercise more of the code by making this an indexed
    # block
    model.p_second_stage = Piecewise([1], model.fz, model.z,
                                     pw_pts=[-10,-5.,0.,5.,10.],
                                     pw_constr_type='EQ',
                                     pw_repn='INC',
                                     f_rule=[0.,0.,-1.,2.+cnt,1.],
                                     force_pw=True)

    return model

def pysp_instance_creation_callback(scenario_name, node_names):

    model = create_instance(scenario_name)

    #
    # SMPS Related Annotations
    #
    model.stoch_rhs = StochasticConstraintBoundsAnnotation()
    # declarations can be blocks to imply that all
    # components (Constraints in this case) should be
    # considered on that block
    model.stoch_rhs.declare(model.p_second_stage)
    model.stoch_rhs.declare(model.r_second_stage)
    model.stoch_matrix = StochasticConstraintBodyAnnotation()
    # exercise more of the code by testing this with an
    # indexed block and a single block
    model.stoch_matrix.declare(model.c_second_stage, variables=[model.r])
    model.stoch_matrix.declare(model.p_second_stage[1])
    model.stoch_objective = StochasticObjectiveAnnotation()

    return model
