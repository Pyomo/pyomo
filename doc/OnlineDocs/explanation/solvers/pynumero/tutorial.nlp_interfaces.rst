NLP Interfaces
==============

Below are examples of using PyNumero's interfaces to ASL for function
and derivative evaluation. More information can be found in the
:mod:`API documentation <pyomo.contrib.pynumero>`.

Relevant imports

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
   >>> import numpy as np

Create a Pyomo model

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(bounds=(-5, None))
   >>> m.y = pyo.Var(initialize=2.5)
   >>> m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
   >>> m.c1 = pyo.Constraint(expr=m.y == (m.x - 1)**2)
   >>> m.c2 = pyo.Constraint(expr=m.y >= pyo.exp(m.x))

Create a :py:class:`pyomo.contrib.pynumero.interfaces.pyomo_nlp.PyomoNLP` instance

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp = PyomoNLP(m)

Get values of primals and duals

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp.get_primals()
   array([0. , 2.5])
   >>> nlp.get_duals()
   array([0., 0.])

Get variable and constraint bounds

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp.primals_lb()
   array([ -5., -inf])
   >>> nlp.primals_ub()
   array([inf, inf])
   >>> nlp.constraints_lb()
   array([  0., -inf])
   >>> nlp.constraints_ub()
   array([0., 0.])

Objective and constraint evaluations

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp.evaluate_objective()
   6.25
   >>> nlp.evaluate_constraints()
   array([ 1.5, -1.5])

Derivative evaluations

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp.evaluate_grad_objective()
   array([0., 5.])
   >>> nlp.evaluate_jacobian()  # doctest: +SKIP
   <2x2 sparse matrix of type '<class 'numpy.float64'>'
       with 4 stored elements in COOrdinate format>
   >>> nlp.evaluate_jacobian().toarray()
   array([[ 2.,  1.],
          [ 1., -1.]])
   >>> nlp.evaluate_hessian_lag().toarray()
   array([[2., 0.],
          [0., 2.]])

Set values of primals and duals

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp.set_primals(np.array([0, 1]))
   >>> nlp.evaluate_constraints()
   array([0., 0.])
   >>> nlp.set_duals(np.array([-2/3, 4/3]))
   >>> nlp.evaluate_grad_objective() + nlp.evaluate_jacobian().transpose() * nlp.get_duals()
   array([0., 0.])

Equality and inequality constraints separately

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> nlp.evaluate_eq_constraints()
   array([0.])
   >>> nlp.evaluate_jacobian_eq().toarray()
   array([[2., 1.]])
   >>> nlp.evaluate_ineq_constraints()
   array([0.])
   >>> nlp.evaluate_jacobian_ineq().toarray()
   array([[ 1., -1.]])
   >>> nlp.get_duals_eq()
   array([-0.66666667])
   >>> nlp.get_duals_ineq()
   array([1.33333333])
