Multistart Solver
==================

The multistart solver is used in cases where the objective function is known
to be non-convex but the global optimum is still desired. It works by running a non-linear
solver of your choice multiple times at different starting points, and
returns the best of the solutions.


Using Multistart Solver
-----------------------
To use the multistart solver, define your Pyomo model as usual

>>> import pyomo.environ as pe
>>> m = pe.ConcreteModel()
>>> m.x = pe.Var()
>>> m.y = pe.Var()
>>> m.obj = pe.Objective(expr=m.x**2 + m.y**2)
>>> m.c = pe.Constraint(expr=m.y >= -2*m.x + 5)

Instantiate the multistart solver through the SolverFactory

>>> opt = pe.SolverFactory('multistart')

This returns an instance of :py:class:`Multistart`. To solve our Model
we use the solve command

>>> # Keywords:
>>> # 'strategy': specify the restart strategy, defaults to random
>>> #             "rand"
>>> #             "midpoint_guess_and_bound"
>>> #             "rand_guess_and_bound"
>>> #             "rand_distributed"
>>> # 'solver' : specify any solver within the SolverFactory, defaults to ipopt
>>> # 'iterations' : specify the number of iterations, defaults to 10.
>>> #                 if -1 is specified, the high confidence stopping rule will be used
>>> # 'HCS_param' : specify the tuple (m,d)
>>> #               defaults to (m,d) = (.5,.5)
>>> #               only use with random strategy
>>> #               The stopping mass m is the maximum allowable estimated missing mass of optima
>>> #               The stopping delta d = 1-the confidence level required for the stopping rule
>>> #               For both parameters, the lower the parameter the more stricter the rule.
>>> #               both are bounded 0<x<=1
>>> optsolver.solve(m2,iterations = 10);
