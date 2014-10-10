Extended version: there is a singleton var to allow for renting land to the neighbor. Primarily for testing purposes,
i.e., to avoid strictly indexed variables.

This directory contains pyomo/pysp models and data for the "Farmer" stochastic program (and corresponding 
deterministic program), introduced in Section 1.1 of "Introduction to Stochastic Programming" by Birge and 
Louveaux. 

This problem serves as a simple, quick smoke-screen test for a functioning stochastic programming solver.

farmer_mip.py: the deterministic LP model 
farmer_mip.dat: data file for the deterministic LP (expected case parameters)

The deterministic model can be solved simply by the following command-line: pyomo farmer_mip.py farmer_mip.dat

The optimal objective value for the deterministic expected-value model is: -118600 (profot if 118600).
The stochastic farmers problem is split into the following directories:
- models ; contains the scenario tree and scenario reference model files.
- nodedata ; contains node-oriented data for the two-stage case (low, medium, and high yield scenarios).
- scenariodata ; contains scenario-oriented data for the two-stage case. 

The optimal objective value for the stochastic model lis: -108390 (profit of 108390).
The optimal solution for the stochastic model plants 170 acres of wheat, 80 acres of corn, and 250 acres of sugar beets.

The stochasic farmers problem can be solved via the following command-line:

runph --model-directory=models --instance-directory=nodedata --default-rho=1.0 --max-iterations=100

This should converge in 48 iterations, with an expected profit of 108390 (-108390 objective value). 

We did not experiment with any rho values other than 1.0 for this problem - 1.0 works, and we
aren't that interested in solving this problem more quickly. 

To linearize the quadratic penalty terms, simply add the following options to the command line:
 
--linearize-nonbinary-penalty-terms=2 

Four piecewise linear segments per blended variable yields faster convergence (9 iterations), but
a sub-optimal solution (-107958.9).

Fourteen piece linear segments per blended variable yields slow convergence (28 iterations), but
an optimal solution - to within convergence tolerance (-108389).
