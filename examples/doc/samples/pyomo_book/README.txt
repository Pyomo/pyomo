This directory contains examples from the following book:

  Pyomo: Optimization modeling in Python
  William E. Hart, Carl D. Laird, Jean-Paul Watson, David L. Woodruff

------- ------- ------- ------- ------- ------- ------- ------- ------- -------

overview_concrete1.py - A concrete Pyomo model using explicit variables.

overview_concrete2.py - A concrete Pyomo model using indexed variables.

overview_concrete3.py - A concrete Pyomo model with (a) external data declarations and (b) expressions defined using Python's generator syntax.

overview_concrete4.py - A concrete Pyomo model using a constraint rule to create constraints.

overview_concrete5.py - A concrete Pyomo model that uses rule and initialize arguments for all modeling components.

overview_abstract4.py - A function that creates a concrete Pyomo model using only data provide in the argument list.

overview_abstract5.py - An abstract Pyomo model.

overview_abstract5.dat - Data commands for the abstract Pyomo model in overview_abstract5.py.

overview_script1.py - A Python script that creates a concrete Pyomo model 
and performs optimization using the GLPK linear programming solver.

overview_script2.py - A Python script that creates a model instance from an abstract Pyomo model and performs optimization using the GLPK linear programming solver.

miscComponents_concrete.py - A Python script that illustrates the use of error checks and diagnostic output in the construction of a concrete model.

miscComponents_abstract.py - A Python script that illustrates the use of BuildAction and BuildCheck components to define error checks and diagnostic output in an abstract model.

data_abstract5_ns1.dat - Data commands that illustrates the use of namespaces that define collections of data commands that are grouped together.

data_diet1.py - A Pyomo model for the classic diet problem. The goal of this problem is to minimize cost while ensuring that the diet meets certain requirements.

data_diet1.dat - A Pyomo data file that defines data for a simple diet problem.

data_diet1.db.dat - A Pyomo data file that imports data from the Access data base file diet.mdb.

command_abstract6.py - An abstract Pyomo model that is stored in the Model variable.

command_abstract5_ns2.dat - Data commands that illustrate the use of
namespaces that define collections of data commands that are grouped together.

nonlinear_Rosenbrock.py - A Pyomo model for the Rosenbrock function.

nonlinear_multimodal_init1.py - A Pyomo model that defines a simple multimodal test problem.

nonlinear_DeerProblem.py - An abstract Pyomo model for the formulation of the optimal sustainable deer harvesting problem.

nonlinear_DeerProblem.dat - The data file for the sustainable deer harvesting problem.

nonlinear_DiseaseEstimation.py - A Pyomo model for disease estimation.

nonlinear_DiseaseEstimation.dat - A Pyomo data file that contains the set definitions, the population, and the case count data for disease estimation.

nonlinear_ReactorDesign.py - A concrete Pyomo model for a simple reactor design problem.

pysp_farmerReferenceModel.py - The deterministic reference Pyomo model for Birge and Louveaux's farmer problem.

pysp_farmerReferenceModel.dat - The Pyomo reference model data for Birge and Louveaux's farmer problem.

pysp_farmerScenarioStructure.dat - The data command file that defines the
scenario tree for the farmer problem.

scripts_script1.py - A Python script that creates an instance of a Pyomo model and performs optimization using the IPOPT solver.

scripts_script2.py - A Python script that creates an instance of a Pyomo model using two different data files, and performs optimization using the IPOPT solver.

scripts_minimalistic.py - A Python script that creates an instance of a Pyomo model, provides a detailed control of the optimization process.

scripts_mimiPyomo.py - A Python script that creates an instance of a Pyomo model and performs optimization using the IPOPT solver. This script leverages Pyomo scripting functionality to provide a high-level of control over the optimization process. 

scripts_script1vars.py - A Python script that creates two concrete Pyomo models, performs optimization using the GLPK linear programming solver, and then compares the value of model variables. 

scripts_scriptacrossvars.py - A Python script that creates a concrete Pyomo model, performs optimization using the GLPK linear programming solver and then outputs the values of all variables.  

scripts_multimodal_multiinit.py - A Python script that finds and prints the solution for two starting different starting     points.

scripts_multimodal_gridinit.py - A Python script that explores a grid of starting points and keeps a list of all of the     unique solutions that were found.

scripts_script2fix_all.py - A Pyomo script that creates a nonlinear multimodal problem, performs optimization with IPOPT, and reoptimizes the problem with a fixed    variable.

scripts_DiseaseAddDrop.py - A Pyomo script that defines a model with both hard and easy constraints.  A model with the easy constraints is solved first, and that solution is used to initialize the optimizer when solving with the hard constraints.

scripts_DiseaseEasy.py - A Pyomo disease model that contains only the easy constraints.

scripts_run_mult_models.py - A Pyomo script that shares results between two models, DiseaseEasy and DiseaseHard.

scripts_DiseasePlot.py - A Pyomo script that optimizes a disease model and plots the estimated data versus the measured data.

scripts_sudoku.py - A Pyomo script that defines a function that    returns a Pyomo model for a sudoku problem.

scripts_sudoku_run.py - A Pyomo script that iteratively adds cuts for solutions that have been found.

scripts_hybrid_main.py - A Pyomo script that iterates between two different optimizers to find global optima.

