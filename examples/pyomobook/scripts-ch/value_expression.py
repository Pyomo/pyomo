import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.u = pyo.Var(initialize=2.0)

# unexpected expression instead of value
a = model.u - 1
print(a)       # "u - 1"
print(type(a)) # <class 'pyomo.core.expr.numeric_expr.SumExpression'>

# correct way to access the value
b = pyo.value(model.u) - 1
print(b) # 1.0 
print(type(b)) # <class 'float'>



