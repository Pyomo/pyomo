import pyomo.modeling
from pyomo.core import *
from pyomo.bilevel import *

model = AbstractModel()

model.V = Param(within=PositiveIntegers)
model.S = Param(within=PositiveIntegers)
model.r0 = Param(within=PositiveIntegers)
model.r = Param(model.VERTICES)

model.VERTICES = RangeSet(1, model.V)
model.SCENARIOS = RangeSet(1, model.S)
model.SCENARIO_VERTICES = Set(within=model.SCENARIOS * model.VERTICES)
model.s = Param(model.SCENARIO_VERTICES)
model.v = Param(model.SCENARIO_VERTICES)
model.p = Param(model.SCENARIO_VERTICES)
model.theta = Param(model.SCENARIOS)
model.x_tilde = Param(model.VERTICES)

# define variables
model.d=Var(model.VERTICES, within=Binary)
model.w=Var(model.SCENARIO_VERTICES, within=NonNegativeReals)

# define objective function
def obj_rule(model):
    M = model.model()
    return sum(M.theta[sv[0]] * p[sv] * w[sv] for sv in model.SCENARIO_VERTICES)
model.obj = Objective(rule=obj_rule, sense=maximize)

# sensor interdiction budget constraint
def budget_rule(model):
    return sum(model.r[v] * model.d[v] for v in model.VERTICES) <= model.r0
model.budget = Constraint(rule=budget_rule)

# define the submodel
model.sub = SubModel(fixed=model.d)
# define objective function with the same expression but opposite sense
model.sub.obj = Objective(rule=obj_rule, sense=minimize)

# define first sensor to detect constraints
def _alpha(model, sv):
    M = model.model()
    return sum(M.w[sv] for sv in M.SCENARIO_VERTICES if sv[0] == s) == 1
model.sub.alpha = Constraint(model.SCENARIOS, rule=_alpha)

# define sensor failure constraints
def _beta(model, sv):
    M = model.model()
    return M.w[sv] <= M.x_tilde[sv[1]] * (1-M.d[sv[1]])
model.sub.beta = Constraint(model.SCENARIO_VERTICES, rule=_beta)

# define w upper-bound constraints
def _gamma(model, sv):
    M = model.model()
    return M.w[sv] <= 1
model.sub.gamma = Constraint(model.SCENARIO_VERTICES, rule=_gamma)

