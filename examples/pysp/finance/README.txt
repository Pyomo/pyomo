This directory contains pyomo/pysp models and data for the "Financial Planning" stochastic
program (and corresponding deterministic program), introduced in Section 1.2 of "Introduction
to Stochastic Programming" by Birge and Louveaux.

Building on the farmer example, this problem is still relatively simple (all continuous variables)
but multi-stage. Intent is to exercise some multi-stage aspects of stochastic programming solvers,
before diving into multi-stage plus integers.

financial_planning.py:  the deterministic LP model.
financial_planning.dat: data file for the deterministic LP (expected-case parameters).

The deterministic model can be solved simply by the following command-line: pyomo solve financial_planning.py financial_planning.dat

the objective function value is 4743.94

The stochastic model can be solved with a command line like:

runef -i scenariodata/ -m models --solve --solver=gurobi

The solution to the stochastic version has objective function value (expected code) $-1514
