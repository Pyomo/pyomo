PyNumero Interfaces 
=============================================

PyNumero provides a number of interfaces to facilitate the development of NLP algorithms. All these interfaces can be found under **pyomo.contrib.pynumero.interfaces**. The overall idea of these interfaces is to provide functionality to query nonlinear projects to get information such as number of variables, number of constraints, upper and lower bounds. More importantly, these interfaces allow users to evaluate NLP functions, including objective and constraint evaluations as well as jacobian of the constraints and hessian of the lagrangian function. A key feature from these interfaces is that rely on Numpy and Scipy for storing results from NLP evaluations. For instance, the evaluation of constraints in one of these NLP interaces takes a numpy.ndarray and returns a numpy.ndarray. Similarly, for the evaluation of Jacobian of constraints or Hessian of Lagrangian, the interfaces take numpy.ndarrays as input and return Scipy sparse matrices.   

.. automodule:: pyomo.contrib.pynumero.interfaces
    :members:
    :no-undoc-members:
    :show-inheritance:

Submodules
----------

.. toctree::

   pynumero.interfaces.nlp
   pynumero.interfaces.amplnlp
   pynumero.interfaces.pyomonlp
   pynumero.interfaces.nlp_compositions
   pynumero.interfaces.nlp_state
