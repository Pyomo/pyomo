.. _examplesection:

Examples
========

Examples can be found in `pyomo/contrib/parmest/examples` and include:

* Reactor design example [PyomoBookII]_
* Semibatch example [SemiBatch]_
* Rooney Biegler example [RooneyBiegler]_

Each example contains a Python file that contains the Pyomo model and a Python file to run parameter estimation.

The description below uses the reactor design example.
The file **reactor_design.py** includes a function which returns an populated instance of the Pyomo model.  
Note that the model is defined to maximize `cb` and that `k1`, `k2`, and `k3` are fixed.  
The _main_ program is included for easy testing of the model declaration.

.. literalinclude:: ../../../../pyomo/contrib/parmest/examples/reactor_design/reactor_design.py
   :language: python

The file **reactor_design_parmest.py** uses parmest to estimate values of `k1`, `k2`, and `k3` by minimizing the sum of 
squared error between model and observed values of `ca`, `cb`, `cc`, and `cd`.  The file also uses parmest to 
run parameter estimation with bootstrap resampling and perform a likelihood ratio test over a range of 
theta values.

.. literalinclude:: ../../../../pyomo/contrib/parmest/examples/reactor_design/reactor_design_parmest.py
   :language: python

Additional examples are defined in a similar manner.  
The reactor design example includes an additional data file with duplicate sensor data and 
the semibatch example includes an example which can be used to run parmest in parallel.