"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for pysp.rst in testable form
"""
from pyomo.environ import *
model = ConcreteModel()
p=1.0
q=1.0
# @Import&declare_annotations
from pyomo.pysp.annotations import *
model.stoch_rhs = StochasticConstraintBoundsAnnotation()
model.stoch_matrix = StochasticConstraintBodyAnnotation()
model.stoch_objective = StochasticObjectiveAnnotation()
# @Import&declare_annotations

# @Partially_declared_concretemodel
model = ConcreteModel()

# data that is initialized on a per-scenario basis
#p = ...
#q = ...

# variables declared as second-stage on the
# PySP scenario tree
model.z = Var()
model.y = Var()

# indexed constraint
model.r_index = Set(initialize=[3, 6, 9])
def r_rule(model, i):
    return p + i <= 1 *model.z + model.y * 5 <= 10 + q + i
model.r = Constraint(model.r_index, rule=r_rule)

# singleton constraint
model.c = Constraint(expr= p * model.z >= 1)

# a sub-block with a singleton constraint
model.b = Block()
model.b.c = Constraint(expr= q * model.y >= 1)
# @Partially_declared_concretemodel

# @Model_implicit_form
model.stoch_rhs = StochasticConstraintBoundsAnnotation()
# @Model_implicit_form

# @Model_implicit_block
model.stoch_rhs = StochasticConstraintBoundsAnnotation()
model.stoch_rhs.declare(model)
# @Model_implicit_block

# @Explicit_singletonconstraint_implicit_indexedconstraint&subblock
model.stoch_rhs = StochasticConstraintBoundsAnnotation()
model.stoch_rhs.declare(model.r)
model.stoch_rhs.declare(model.c)
model.stoch_rhs.declare(model.b)
# @Explicit_singletonconstraint_implicit_indexedconstraint&subblock

# @Explicit_singletonconstraint&subblock_implicit_indexedconstraint
model.stoch_rhs = StochasticConstraintBoundsAnnotation()
model.stoch_rhs.declare(model.r)
model.stoch_rhs.declare(model.c)
model.stoch_rhs.declare(model.b.c)
# @Explicit_singletonconstraint&subblock_implicit_indexedconstraint

# @Explicit_singletonconstraint_explicit_indexedconstraint
model.stoch_rhs = StochasticConstraintBoundsAnnotation()
model.stoch_rhs.declare(model.r[3])
model.stoch_rhs.declare(model.r[6])
model.stoch_rhs.declare(model.r[9])
model.stoch_rhs.declare(model.c)
model.stoch_rhs.declare(model.b.c)
# @Explicit_singletonconstraint_explicit_indexedconstraint

# @Annotation_Abstractmodel
def annotate_rule(m):
    m.stoch_rhs = StochasticConstraintBoundsAnnotation()
    m.stoch_rhs.declare(m.r[3])
    m.stoch_rhs.declare(m.r[6])
    m.stoch_rhs.declare(m.r[9])
    m.stoch_rhs.declare(m.c)
    m.stoch_rhs.declare(m.b.c)
model.annotate = BuildAction(rule=annotate_rule)
# @Annotation_Abstractmodel

p=1
q=1
# @Stochastic_constraint_bounds
from pyomo.pysp.annotations import StochasticConstraintBoundsAnnotation

model = ConcreteModel()

# data that is initialized on a per-scenario basis
#p = ...
#q = ...

# a second-stage variable
model.y = Var()

# declare the annotation
model.stoch_rhs = StochasticConstraintBoundsAnnotation()

# equality constraint
model.c = Constraint(expr= model.y == q)
model.stoch_rhs.declare(model.c)

# double-sided inequality constraint with
# stochastic upper bound
model.r = Constraint(expr= 0 <= model.y <= p)
model.stoch_rhs.declare(model.r, lb=False)

# indexed constraint using a BuildAction
model.C_index = RangeSet(1,3)
def C_rule(model, i):
    if i == 1:
        return model.y >= i * q
    else:
        return Constraint.Skip
model.C = Constraint(model.C_index, rule=C_rule)
def C_annotate_rule(model, i):
    if i == 1:
        model.stoch_rhs.declare(model.C[i])
    else:
        pass
model.C_annotate = BuildAction(model.C_index, rule=C_annotate_rule)
# @Stochastic_constraint_bounds

p=1
q=1
# @Stochastic_constraint_matrix
from pyomo.pysp.annotations import StochasticConstraintBodyAnnotation

model = ConcreteModel()

# data that is initialized on a per-scenario basis
#p = ...
#q = ...

# a first-stage variable
model.x = Var()

# a second-stage variable
model.y = Var()

# declare the annotation
model.stoch_matrix = StochasticConstraintBodyAnnotation()

# a singleton constraint with stochastic coefficients
# both the first- and second-stage variable
model.c = Constraint(expr= p * model.x + q * model.y == 1)
model.stoch_matrix.declare(model.c)
# an assignment that is equivalent to the previous one
model.stoch_matrix.declare(model.c, variables=[model.x, model.y])

# a singleton range constraint with a stochastic coefficient
# for the first-stage variable only
model.r = Constraint(expr= 0 <= p * model.x - 2.0 * model.y <= 10)
model.stoch_matrix.declare(model.r, variables=[model.x])
# @Stochastic_constraint_matrix

p=1
q=1
# @Stochastic_objective_elements
from pyomo.pysp.annotations import StochasticObjectiveAnnotation

model = ConcreteModel()

# data that is initialized on a per-scenario basis
#p = ...
#q = ...

# a first-stage variable
model.x = Var()

# a second-stage variable
model.y = Var()

# declare the annotation
model.stoch_objective = StochasticObjectiveAnnotation()

model.FirstStageCost = Expression(expr= 5.0 * model.x)
model.SecondStageCost = Expression(expr= p * model.x + q * model.y)
model.TotalCost = Objective(expr= model.FirstStageCost + model.SecondStageCost)

# each of these declarations is equivalent for this model
model.stoch_objective.declare(model.TotalCost)
model.stoch_objective.declare(model.TotalCost, variables=[model.x, model.y])
# @Stochastic_objective_elements

"""
xxxxxxxxxxxxxxxxxxxxxx alert!!!! as May 2018 not testing about half the file xxxxxxxxxxxxxxxxxx
# @Annotating_constraint_stages
from pyomo.pysp.annotations import PySP_ConstraintStageAnnotation()

# declare the annotation
model.constraint_stage = PySP_ConstraintStageAnnotation()

# all constraints on this Block are first-stage
model.B = Block()
...
model.constraint_stage.declare(model.B, 1)

# all indices of this indexed constraint are first-stage
model.C1 = Constraint(..., rule=...)
model.constraint_stage.declare(model.C1, 1)

# all but one index in this indexed constraint are second-stage
model.C2 = Constraint(..., rule=...)
for index in model.C2:
    if index == 'a':
        model.constraint_stage.declare(model.C2[index], 1)
    else:
        model.constraint_stage.declare(model.C2[index], 2)
# @Annotating_constraint_stages

# @1&2stage_in_2stage_expression
from pyomo.pysp.annotations import StochasticObjectiveAnnotation

model = ConcreteModel()

# data that is initialized on a per-scenario basis
p = ...
q = ...

# first-stage variable
model.x = Var()
# second-stage variable
model.y = Var()

# first-stage cost expression
model.FirstStageCost = Expression(expr= 5.0 * model.x)
# second-stage cost expression
model.SecondStageCost = Expression(expr= p * model.x + q * model.y)

# define the objective as the sum of the
# stage-cost expressions
model.TotalCost = Objective(expr= model.FirstStageCost + model.SecondStageCost)

# declare that model.x and model.y have stochastic cost
# coefficients in the second stage
model.stoch_objective = StochasticObjectiveAnnotation()
model.stoch_objective.declare(model.TotalCost, variables=[model.x, model.y])
# @1&2stage_in_2stage_expression

# @Reexpressed_model_using_new_variable
from pyomo.pysp.annotations import StochasticConstraintBodyAnnotation

model = ConcreteModel()

# data that is initialized on a per-scenario basis
p = ...
q = ...

# first-stage variable
model.x = Var()
# second-stage variables
model.y = Var()
model.SecondStageCostVar = Var()

# first-stage cost expression
model.FirstStageCost = Expression(expr= 5.0 * model.x)
# second-stage cost expression
model.SecondStageCost = Expression(expr= p * model.x + q * model.y)

# define the objective using SecondStageCostVar
# in place of SecondStageCost
model.TotalCost = Objective(expr= model.FirstStageCost + model.SecondStageCostVar)

# set the variable SecondStageCostVar equal to the
# expression SecondStageCost using an equality constraint
model.ComputeSecondStageCost = Constraint(expr= model.SecondStageCostVar == model.SecondStageCost)

# declare that model.x and model.y have stochastic constraint matrix
# coefficients in the ComputeSecondStageCost constraint
model.stoch_matrix = StochasticConstraintBodyAnnotation()
model.stoch_matrix.declare(model.ComputeSecondStageCost, variables=[model.x, model.y])
# @Reexpressed_model_using_new_variable

# @Constraint_in_PySP_StochasticRHSAnnotation
model = AbstractModel()

# a first-stage variable
model.x = Var()

# a second-stage variable
model.y = Var()

# a param initialized with scenario-specific data
model.p = Param()

# a second-stage constraint with a stochastic upper bound
# hidden in the left-hand-side expression
def c_rule(m):
    return (m.x - m.p) + m.y <= 10
model.c = Constraint(rule=c_rule)
# @Constraint_in_PySP_StochasticRHSAnnotation


# @Variable_excluded_in_subsetscenerios
model = ConcreteModel()

# data that is initialized on a per-scenario basis
# with q set to zero for this particular scenario
p = ...
q = 0

model.x = Var()
model.y = Var()

model.c1 = Constraint(expr= p * model.x + q * model.y == 1)

def c2_rule(model):
    expr = p * model.x
    if q != 0:
       expr += model.y
    return expr >= 0
model.c2 = Constraint(rule=c2_rule)
# @Variable_excluded_in_subsetscenerios

# @Create_zero_expression_object
model.zero = Expression(expr=0)
# @Create_zero_expression_object


# @Retain_variable_reference_expression
from pyomo.environ import *
model = ConcreteModel()
model.x = Var()
model.y = Var()
model.zero = Expression(expr=0)

# an expression that does NOT
# retain model.y
print((model.x + 0 * model.y).to_string())               # -> x

# an equivalent expression that DOES
# retain model.y
print((model.x + model.zero * model.y).to_string())      # -> x + 0.0 * y

# an equivalent expression that does NOT
# retain model.y (so beware)
print((model.x + 0 * model.zero * model.y).to_string())  # -> x
# @Retain_variable_reference_expression
"""
