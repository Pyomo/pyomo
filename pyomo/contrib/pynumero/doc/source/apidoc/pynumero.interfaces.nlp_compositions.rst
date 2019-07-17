Composite NLPs
==============

These interfaces are examples of NLP problems that take advantage of structure to evaluate NLP functions in a decomposable fashion. We present here two examples. The first example is an interface for a two-stage stochatic optimization problem and the second is an interface for a nonlinear program that results from discretizing the time domain in a dynamic optimization problem. Both examples demonstrate the power of using the BlockVector and BlockMatrix objects in **pyomo.contrib.pynumero.sparse**. 

TwoStageStochasticNLP Class
###########################

.. autoclass:: pyomo.contrib.pynumero.interfaces.nlp_compositions.TwoStageStochasticNLP
   :members:
   :inherited-members:
   :show-inheritance:

DynoptNLP Class
########################
      
.. autoclass:: pyomo.contrib.pynumero.interfaces.nlp_compositions.DynoptNLP
   :members:
   :inherited-members:
   :show-inheritance:
