#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import ConcreteModel, Param, Var, Objective, Constraint, Expression, Piecewise, sum_product
from pyomo.pysp.embeddedsp import (EmbeddedSP,
                                   StochasticDataAnnotation,
                                   TableDistribution,
                                   StageCostAnnotation,
                                   VariableStageAnnotation)

def create_embedded():

    model =  ConcreteModel()
    model.d1 =  Param(mutable=True, initialize=0)
    model.d2 =  Param(mutable=True, initialize=0)
    model.d3 =  Param(mutable=True, initialize=0)
    model.d4 =  Param(mutable=True, initialize=0)
    # first stage
    model.x =  Var(bounds=(0,10))
    # first stage derived
    model.y =  Expression(expr=model.x + 1)
    model.fx =  Var()
    # second stage
    model.z =  Var(bounds=(-10, 10))
    # second stage derived
    model.q =  Expression(expr=model.z**2)
    model.fz =  Var()
    model.r =  Var()
    # stage costs
    model.StageCost =  Expression([1,2])
    model.StageCost.add(1, model.fx)
    model.StageCost.add(2, -model.fz + model.r + model.d1)
    model.o =  Objective(expr= sum_product(model.StageCost))

    model.c_first_stage =  Constraint(expr= model.x >= 0)

    # test our handling of intermediate variables that
    # are created by Piecewise but can not necessarily
    # be declared on the scenario tree
    model.p_first_stage =  Piecewise(model.fx, model.x,
                                        pw_pts=[0.,2.,5.,7.,10.],
                                        pw_constr_type='EQ',
                                        pw_repn='INC',
                                        f_rule=[10.,10.,9.,10.,10.],
                                        force_pw=True)

    model.c_second_stage =  Constraint(expr= model.x + model.r * model.d2 >= -100)
    model.cL_second_stage =  Constraint(expr= model.d3 >= -model.r)
    model.cU_second_stage =  Constraint(expr= model.r <= 0)

    # exercise more of the code by making this an indexed
    # block
    model.p_second_stage =  Piecewise([1], model.fz, model.z,
                                         pw_pts=[-10,-5.,0.,5.,10.],
                                         pw_constr_type='EQ',
                                         pw_repn='INC',
                                         f_rule=[0.,0.,-1.,model.d4,1.],
                                         force_pw=True)

    # annotate the model
    model.varstage = VariableStageAnnotation()
    # first stage
    model.varstage.declare(model.x, 1)
    model.varstage.declare(model.y, 1, derived=True)
    model.varstage.declare(model.fx, 1, derived=True)
    model.varstage.declare(model.p_first_stage, 1, derived=True)
    # second stage
    model.varstage.declare(model.z, 2)
    model.varstage.declare(model.q, 2, derived=True)
    model.varstage.declare(model.fz, 2, derived=True)
    model.varstage.declare(model.r, 2, derived=True)
    model.varstage.declare(model.p_second_stage, 2, derived=True)

    model.stagecost = StageCostAnnotation()
    for i in [1,2]:
        model.stagecost.declare(model.StageCost[i], i)

    model.stochdata = StochasticDataAnnotation()
    model.stochdata.declare(
        model.d1,
        distribution=TableDistribution([0.0,1.0,2.0]))
    model.stochdata.declare(
        model.d2,
        distribution=TableDistribution([0.0,1.0,2.0]))
    model.stochdata.declare(
        model.d3,
        distribution=TableDistribution([0.0,1.0,2.0]))
    model.stochdata.declare(
        model.d4,
        distribution=TableDistribution([0.0,1.0,2.0]))

    return EmbeddedSP(model)
