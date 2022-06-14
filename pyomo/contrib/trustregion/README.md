# Trust Region Filter Algorithm

The trust region filter algorithm was initially introduced into Pyomo
based on the work by Eason/Biegler in their 2016 and 2018 papers in AIChE.

The algorithm has been updated to match work by Yoshio/Biegler in their
2021 paper in AIChE.

The algorithm, at its core, takes a model and makes incremental steps towards
an optimal solution using a surrogate model.

Full details on the algorithm can be found in:

> Yoshio, N., & Biegler, L. T. (2021). Demand‐based optimization of a chlorobenzene process with high‐fidelity and surrogate reactor models under trust region strategies. AIChE Journal, 67(1), e17054.

## Assumptions

Several assumptions are made about the model:

1. The objective function, equality constraints, and inequality constraints are assumed to be twice-differentiable.
2. For external functions on the model, the gradient is also supplied.
3. All variables are appropriately scaled.

## Usage

The trust region filter algorithm can be used like a normal 'solver':

```
solver = SolverFactory('trustregion')
```

To run the algorithm, call the `solve` method:

```
solver.solve(...)
```

The arguments required for the `solve` method are:

1. `model` : The model to be solved
2. `degrees_of_freedom_variables` : A list of variables representing the degrees of freedom within the model

Optionally, the user can also supply `ext_fcn_surrogate_map_rule`, which is the
low-fidelity model used as the 'basis function' `b(w)` in the surrogate model. 
Examples with and without `ext_fcn_surrogate_map_rule` can be found in 
the `examples` directory.

A sample model is shown below:

```
from pyomo.environ import (
    ConcreteModel, Var, Reals, ExternalFunction, sin, cos,
    sqrt, Constraint, Objective)
from pyomo.opt import SolverFactory

m = ConcreteModel()
m.z = Var(range(3), domain=Reals, initialize=2.)
m.x = Var(range(2), initialize=2.)
m.x[1] = 1.0

def blackbox(a, b):
   return sin(a - b)

def grad_blackbox(args, fixed):
    a, b = args[:2]
    return [ cos(a - b), -cos(a - b) ]

m.ext_fcn = ExternalFunction(blackbox, grad_blackbox)

m.obj = Objective(
    expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
       + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6
)

m.c1 = Constraint(
    expr=m.x[0] * m.z[0]**2 + m.ext_fcn(m.x[0], m.x[1]) == 2*sqrt(2.0)
    )
m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

optTRF = SolverFactory('trustregion', maximum_iterations=10, verbose=True)
optTRF.solve(m, [m.z[0], m.z[1], m.z[2]])
```
