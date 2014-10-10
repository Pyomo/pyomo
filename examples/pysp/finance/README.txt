This directory contains pyomo/pysp models and data for the "Financial Planning" stochastic
program (and corresponding deterministic program), introduced in Section 1.2 of "Introduction
to Stochastic Programming" by Birge and Louveaux.

Building on the farmer example, this problem is still relatively simple (all continuous variables)
but multi-stage. Intent is to exercise some multi-stage aspects of stochastic programming solvers,
before diving into multi-stage plus integers.

financial_planning.py:  the deterministic LP model.
financial_planning.dat: data file for the deterministic LP (expected-case parameters).

The deterministic model can be solved simply by the following command-line: pyomo financial_planning.py financial_planning.dat

The optimal objective value for the deterministic expected-value model is 4743.938125. The optimal
solution puts all investments into stocks, which is correct given the greater expected return.
DEBUG: The objective value, however, is *not* correct - at least relative to Birge/Louveaux.
DEBUG: THE AMOUNTS INVESTED ARE CORRECT, AS IS THE GROWTH IN $$$. 
DEBUG: WE END UP WITH A TARGET SURPLUS, WHICH IS WRONG - WE HAVE A DEFICIT!




