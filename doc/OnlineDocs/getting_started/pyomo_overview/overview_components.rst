Overview of Modeling Components and Processes
---------------------------------------------

Pyomo supports an object-oriented design for the definition of
optimization models.  The basic steps of a simple modeling process are:

* Create model and declare components
* Instantiate the model
* Apply solver
* Interrogate solver results

In practice, these steps may be applied repeatedly with different data
or with different constraints applied to the model.  However, we focus
on this simple modeling process to illustrate different strategies for
modeling with Pyomo.

A Pyomo *model* consists of a collection of modeling *components* that
define different aspects of the model.  Pyomo includes the modeling
components that are commonly supported by modern AMLs: index sets,
symbolic parameters, decision variables, objectives, and constraints.
These modeling components are defined in Pyomo through the following
Python classes:

Set
***
  set data that is used to define a model instance

Param
*****
  parameter data that is used to define a model instance

Var
***
  decision variables in a model

Objective
*********
  expressions that are minimized or maximized in a model

Constraint
**********
  constraint expressions that impose restrictions on variable values in a model
