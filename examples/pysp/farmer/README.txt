This directory contains pyomo/pysp models and data for the "Farmer" stochastic program (and corresponding 
deterministic program), introduced in Section 1.1 of "Introduction to Stochastic Programming" by Birge and 
Louveaux. 

This problem serves as a simple, quick smoke-screen test for a functioning stochastic programming solver.

The deterministic, expected-value model can be solved simply by the following command-line: 

pyomo solve --solver=cplex models/ReferenceModel.py scenariodata/AverageScenario.dat

The optimal objective value for the deterministic expected-value model is: -118600 (profit if 118600).

The stochastic farmers problem is split into the following directories:
- models ; contains the scenario tree and scenario reference model files.
- nodedata ; contains node-oriented data for the two-stage case (low, medium, and high yield scenarios).
- scenariodata ; contains scenario-oriented data for the two-stage case. 

The optimal objective value for the stochastic model is: -108390 (profit of 108390).
The optimal solution for the stochastic model plants 170 acres of wheat, 80 acres of corn, and 250 acres of sugar beets.

The stochasic farmers problem can be solved via the following command-line:

runph --model-directory=models --instance-directory=nodedata --default-rho=1

This should converge in 48 iterations, with an expected profit of 108390 (-108390 objective value). 

We did not experiment with any rho values other than 1.0 for this problem - 1.0 works, and we
aren't that interested in solving this problem more quickly. 

To linearize the quadratic penalty terms, simply add the following options to the command line:
 
--linearize-nonbinary-penalty-terms=2 

Four piecewise linear segments per blended variable yields faster convergence (9 iterations), but
a sub-optimal solution (-107958.9).

Fourteen piece linear segments per blended variable yields slow convergence (28 iterations), but
an optimal solution - to within convergence tolerance (-108389).

All of the above solutions were obtained with CPLEX 12.2 - different solvers may yield slightly
different convergence behaviors.

You can also run this example using MPI, leveraging PySP's PH solver servers. An example 
command-line is as follows:

mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory=models --instance-directory=scenariodata --default-rho=1.0
