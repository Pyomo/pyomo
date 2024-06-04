Scenario Creation
=================

In addition to model-based parameter estimation, parmest can create
scenarios for use in optimization under uncertainty. To do this, one
first creates an ``Estimator`` object, then a ``ScenarioCreator``
object, which has methods to add ``ParmestScen`` scenario objects to a
``ScenarioSet`` object, which can write them to a csv file or output them
via an iterator method.

This example is in the semibatch subdirectory of the examples directory in
the file ``scenario_example.py``. It creates a csv file with scenarios that
correspond one-to-one with the experiments used as input data. It also
creates a few scenarios using the bootstrap methods and outputs prints the
scenarios to the screen, accessing them via the ``ScensItator`` a ``print``

.. literalinclude:: ../../../../pyomo/contrib/parmest/examples/semibatch/scenario_example.py
   :language: python

.. note::
   This example may produce an error message if your version of Ipopt is not based
   on a good linear solver.
