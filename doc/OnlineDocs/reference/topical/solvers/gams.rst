GAMS
====

.. currentmodule:: pyomo.solvers.plugins.solvers.GAMS

GAMSShell Solver
----------------

.. autosummary::

   GAMSShell
   GAMSShell.available
   GAMSShell.executable
   GAMSShell.solve
   GAMSShell.version
   GAMSShell.warm_start_capable

GAMSDirect Solver
-----------------

.. autosummary::

   GAMSDirect
   GAMSDirect.available
   GAMSDirect.solve
   GAMSDirect.version
   GAMSDirect.warm_start_capable

.. currentmodule:: pyomo.repn.plugins.gams_writer

GAMS Writer
-----------

This class is most commonly accessed and called upon via
model.write("filename.gms", ...), but is also utilized
by the GAMS solver interfaces.

.. autosummary::

   ProblemWriter_gams
