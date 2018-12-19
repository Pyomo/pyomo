.. _examplesection:

Examples
========

Examples can be found in the subdirectories of `pyomo/contrib/parmest/examples`.

Generally, two Python files are used to run parmest (although it can be
done using one or many):

* Driver: This file contains code that uses parmest to compute parameter estimation and confidence regions.

* Callback: This file contains a PySP callback function that returns a model for one scenario (or experiment).  

The Rooney Biegler example below includes an example where the driver and callback are in the same file.

Rooney Biegler Example
----------------------

This example uses the model and data given in [RooneyBiegler]_. 

Callback
^^^^^^^^

The callback file includes:

* Function to generate the model from data. This example defines the model with the objective function
  Expressions named FirstStageCost and SecondStage cost respectively. 
* Callback function, which returns a model for one experiment.

The main program is included for easy testing of the model declaration.

.. literalinclude:: ../../../../pyomo/contrib/parmest/examples/rooney_biegler/rooney_biegler.py
   :language: python
   
Driver
^^^^^^

The driver file creates a :class:`~pyomo.contrib.parmest.parmest.ParmEstimator` object using the file that contains the callback
and uses its methods for parameter estimation.

.. literalinclude::  ../../../../pyomo/contrib/parmest/examples/rooney_biegler/rb_drive_parmest.py
   :language: python

.. _AllInOne:
	      
All in One
^^^^^^^^^^

This example illustrates combining the model, callback and driver in
one file as well as supplying the callback function to the
parmest object constructor instead of strings for the module
and function name. (It is not shown here, but it is also
possible to supply a module and a function name as a string). Note
that this example uses the PySP callback signature, while the example
above uses the parmest callback signature.

.. literalinclude::  ../../../../pyomo/contrib/parmest/examples/rooney_biegler/all_in_one.py
   :language: python

	      
Semibatch Example
-----------------

The semibatch example was created from [SemiBatch]_ by Bethany Nicholson.

Callback
^^^^^^^^

The callback file includes:

* Function to generate the model from data
* Callback function, which loads data from text files

The main program is included for easy testing of the model declaration.

.. literalinclude::  ../../../../pyomo/contrib/parmest/examples/semibatch/semibatch.py
   :language: python
   
Driver
^^^^^^

The driver file creates a class:`~pyomo.contrib.parmest.parmest.ParmEstimator` object using the file that contains the callback
and uses its methods for parameter estimation.

.. literalinclude::  ../../../../pyomo/contrib/parmest/examples/semibatch/sb_drive_parmest.py
   :language: python
   
MEA Example
-----------

MEA is an IDAES model.

Callback
^^^^^^^^

.. literalinclude::  ../../../../pyomo/contrib/parmest/examples/mea/mea_estimate_pysp.py
   :language: python

Driver
^^^^^^

.. literalinclude::  ../../../../pyomo/contrib/parmest/examples/mea/mea_drive_parmest.py
   :language: python
