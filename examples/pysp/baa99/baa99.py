#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# ba99: Annotated with location of stochastic rhs entries
#       for use with pysp2smps conversion tool.

import itertools

from pyomo.core import *
from pyomo.pysp.annotations import (PySP_ConstraintStageAnnotation,
                                    PySP_StochasticRHSAnnotation)

#
# Define the probability table for the stochastic parameters
#

d1_rhs_table = \
    [17.75731865,
     32.96224832,
     43.68044355,
     52.29173734,
     59.67893765,
     66.27551249,
     72.33076402,
     78.00434172,
     83.40733268,
     88.62275117,
     93.71693266,
     98.74655459,
     103.7634931,
     108.8187082,
     113.9659517,
     119.2660233,
     124.7925174,
     130.6406496,
     136.9423425,
     143.8948148,
     151.8216695,
     161.326406,
     173.7895514,
     194.0396804,
     216.3173937]

d2_rhs_table = \
    [5.960319592,
     26.21044859,
     38.673594,
     48.17833053,
     56.10518525,
     63.05765754,
     69.35935045,
     75.20748263,
     80.73397668,
     86.03404828,
     91.18129176,
     96.2365069,
     101.2534454,
     106.2830673,
     111.3772488,
     116.5926673,
     121.9956583,
     127.669236,
     133.7244875,
     140.3210624,
     147.7082627,
     156.3195565,
     167.0377517,
     182.2426813,
     216.3173937]

num_scenarios = len(d1_rhs_table) * len(d2_rhs_table)
scenario_data = dict(('Scenario'+str(i), (d1val, d2val))
                      for i, (d1val, d2val) in
                     enumerate(itertools.product(d1_rhs_table,
                                                 d2_rhs_table), 1))

#
# Define the reference model
#

model = ConcreteModel()

# these annotations are required for using this
# model with the SMPS conversion tool
model.constraint_stage = PySP_ConstraintStageAnnotation()
model.stoch_rhs = PySP_StochasticRHSAnnotation()

# use mutable parameters so that the constraint
# right-hand-sides can be updated for each scenario
model.d1_rhs = Param(mutable=True, initialize=0.0)
model.d2_rhs = Param(mutable=True, initialize=0.0)

# first-stage variables
model.x1 = Var(bounds=(0,217))
model.x2 = Var(bounds=(0,217))

# second-stage variables
model.v1 = Var(within=NonNegativeReals)
model.v2 = Var(within=NonNegativeReals)
model.u1 = Var(within=NonNegativeReals)
model.u2 = Var(within=NonNegativeReals)
model.w11 = Var(within=NonNegativeReals)
model.w12 = Var(within=NonNegativeReals)
model.w22 = Var(within=NonNegativeReals)

# stage-cost expressions
model.FirstStageCost = \
    Expression(initialize=(4*model.x1 + 2*model.x2))
model.SecondStageCost = \
    Expression(initialize=(-8*model.w11 - 4*model.w12 - 4*model.w22 +\
                           0.2*model.v1 + 0.2*model.v2 + 10*model.u1 + 10*model.u2))

#
# this model only has second-stage constraints
#

model.s1 = Constraint(expr=-model.x1 + model.w11 + model.w12 + model.v1 == 0)
model.constraint_stage.declare(model.s1, 2)

model.s2 = Constraint(expr=-model.x2 + model.w22 + model.v2 == 0)
model.constraint_stage.declare(model.s2, 2)

#
# these two constraints have stochastic right-hand-sides
#

model.d1 = Constraint(expr=model.w11 + model.u1 == model.d1_rhs)
model.constraint_stage.declare(model.d1, 2)
model.stoch_rhs.declare(model.d1)

model.d2 = Constraint(expr=model.w12 + model.w22 + model.u2 == model.d2_rhs)
model.constraint_stage.declare(model.d2, 2)
model.stoch_rhs.declare(model.d2)

# always define the objective as the sum of the stage costs
model.obj = Objective(expr=model.FirstStageCost + model.SecondStageCost)

def pysp_scenario_tree_model_callback():
    from pyomo.pysp.scenariotree.tree_structure_model import \
        CreateConcreteTwoStageScenarioTreeModel

    st_model = CreateConcreteTwoStageScenarioTreeModel(num_scenarios)

    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()

    # First Stage
    st_model.StageCost[first_stage] = 'FirstStageCost'
    st_model.StageVariables[first_stage].add('x1')
    st_model.StageVariables[first_stage].add('x2')

    # Second Stage
    st_model.StageCost[second_stage] = 'SecondStageCost'
    st_model.StageVariables[second_stage].add('v1')
    st_model.StageVariables[second_stage].add('v2')
    st_model.StageVariables[second_stage].add('u1')
    st_model.StageVariables[second_stage].add('u2')
    st_model.StageVariables[second_stage].add('w11')
    st_model.StageVariables[second_stage].add('w12')
    st_model.StageVariables[second_stage].add('w22')

    return st_model

def pysp_instance_creation_callback(scenario_name, node_names):

    #
    # Clone a new instance and update the stochastic
    # parameters from the sampled scenario
    #

    instance = model.clone()

    d1_rhs_val, d2_rhs_val = scenario_data[scenario_name]
    instance.d1_rhs.value = d1_rhs_val
    instance.d2_rhs.value = d2_rhs_val

    return instance
