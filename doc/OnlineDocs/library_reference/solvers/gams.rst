GAMS
====

.. currentmodule:: pyomo.solvers.plugins.solvers.GAMS

GAMSShell Solver
----------------

.. autosummary::

   GAMSShell.available
   GAMSShell.executable
   GAMSShell.solve
   GAMSShell.version
   GAMSShell.warm_start_capable

.. autoclass:: GAMSShell
   :members:

GAMSDirect Solver
-----------------

.. autosummary::

   GAMSDirect.available
   GAMSDirect.solve
   GAMSDirect.version
   GAMSDirect.warm_start_capable

.. autoclass:: GAMSDirect
   :members:

.. currentmodule:: pyomo.repn.plugins.gams_writer

GAMS Writer
-----------

This class is most commonly accessed and called upon via
model.write("filename.gms", ...), but is also utilized
by the GAMS solver interfaces.

.. autoclass:: ProblemWriter_gams
   :members: __call__
