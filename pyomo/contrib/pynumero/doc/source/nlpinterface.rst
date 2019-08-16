Nonlinear Programming Problem
=============================

PyNumero considers general nonlinear programming problems of the form:

.. math::
   \begin{eqnarray}
   \textrm{min} & f(x)\nonumber\\
   &g_L\leq g(x)\leq g_U\\
   &x_L\leq x\leq x_U \nonumber
   \end{eqnarray}

where :math:`x\in\Re^{n}` are the primal variables with lower and upper
bounds :math:`x_L\in\Re^{n}, x_U\in\Re^{n}`.The inequality constraints
:math:`g:\Re^{n}\rightarrow\Re^{m}` are bounded by
:math:`g_L\in\Re^{{m}}` and :math:`g_U\in\Re^{{m}}`. We denote the
corresponding dual variables as :math:`y\in\Re^{m}`. PyNumero also
provides explicit distinction between the equality (defined with
:math:`g_L=g_U`) and inequality constraints (defined with :math:`g_L\neq
g_U`) to facilitate the implementation of algorithms that require such
distinction,

.. math::

   \begin{eqnarray}
   \textrm{min} & f(x)\nonumber\\
   \textrm{s.t.}&c(x) = 0\\
   &d_L\leq d(x)\leq d_U\nonumber\\
   &x_L\leq x\leq x_U \nonumber
   \end{eqnarray}

The equality constraints are represented by
:math:`c:\Re^{n}\rightarrow\Re^{m_c}` and
:math:`d:\Re^{n}\rightarrow\Re^{m_d}` denotes the inequality constraints
with bounds :math:`d_L\in\Re^{m_d}` and :math:`d_U\in\Re^{m_d}` and
:math:`m=m_c + m_d`. The PyNumero :py:class:`PyomoNLP
<pyomo.contrib.pynumero.interfaces.nlp.PyomoNLP>` class takes a Pyomo
model and maps it to the general forms defined above. The following
block of code demonstrates some basic features:

.. code-block:: python

	from pyomo.contrib.pynumero.interfaces import PyomoNLP
	from pyomo.contrib.pynumero.sparse import BlockMatrix
	import pyomo.environ as aml

	# define optimization model
	model = aml.ConcreteModel()
	model.x = aml.Var([1, 2, 3], bounds=(0.0, None), initialize=4.0)
	model.c = aml.Constraint(expr=model.x[3] ** 2 + model.x[1] == 25)
	model.d = aml.Constraint(expr=model.x[2] ** 2 + model.x[1] <= 18.0)
	model.o = aml.Objective(expr=model.x[1] ** 4 - 3 * model.x[1]*model.x[2] ** 3 + model.x[3] ** 2 - 8.0)

	# create NLP
	nlp = PyomoNLP(model)
	# initial guesses for primal and dual variables
	x = nlp.x_init()  # array([4., 4., 4.])
	y = nlp.y_init()  # array([0., 0.])
	# variable bounds
	xl = nlp.xl()
	xu = nlp.xu()
	# NLP function evaluations
	f = nlp.objective(x)
	print("Objective Function\n", f)  # -504.0
	g = nlp.evaluate_g(x)
	print("Constraints\n", g)  # array([-5., 20.])
	c = nlp.evaluate_c(x)
	print("Equality Constraints\n", c)  # array([-5.])
	d = nlp.evaluate_d(x)
	print("Inequality Constraints\n", d)  # array([20.])

Evaluation of First and Second-order Derivatives
************************************************

Gradient-based optimization algorithms have been proven to be among the
most efficient algorithms for solving nonlinear optimization
problems. The development of fast automatic differentiation tools makes
it easy to compute derivative information efficiently. State-of-the-art
optimization algorithms typically use first and second-order derivatives
along with numerical linear algebra subroutines to find descent
directions that lead towards an optimal solution. PyNumero uses the Ampl
Solver Library (ASL) to compute derivative information for optimization
problems. The ASL is a collection of C subroutines that compute exact
derivatives efficiently. The `Ctypes
<https://docs.python.org/3/library/ctypes.html>`_ Python package is used
to call the ASL subroutines from Python. PyNumero stores derivative
values from the ASL in Numpy arrays and Scipy sparse matrices. This
leverages the capabilities within the Numpy ecosystem to avoid
marshalling of data between the C and Python environments and enables
performant Python implementations of gradient-based nonlinear
optimization algorithms. It should be noted that even though the current
version of PyNumero only supports calculation of derivatives from the
ASL, interfacing function evaluations of Pyomo models with Numpy arrays
opens a number of possibilities for interfacing with other AD tools
(e.g. PyAdolC). The following code shows how first and second
derivatives can be obtained from a :py:class:`PyomoNLP
<pyomo.contrib.pynumero.interfaces.nlp.PyomoNLP>` instance.

.. code-block:: python

	
	# NLP first and second-order derivatives
	df = nlp.grad_objective(x)
	print("Gradient of Objective Function:\n", df)  # array([-576., 8., 64.])
	jac_g = nlp.jacobian_g(x)
	print("Jacobian of Constraints:\n", jac_g.toarray())  # array([[0., 8., 1.], [8., 0., 1.]])
	jac_c = nlp.jacobian_c(x)
	print("Jacobian of Equality Constraints:\n", jac_c.toarray())  # array([[0., 8., 1.]])
	jac_d = nlp.jacobian_d(x)
	print("Jacobian of Inequality Constraints:\n", jac_d.toarray())  # array([[8., 0., 1.]])
	hess_lag = nlp.hessian_lag(x, y)
	print("Hessian of Lagrangian\n", hess_lag.toarray())  # array[[-288.,   0., -144.],
	                                                      #       [   0.,   2.,    0.],
		                                              #       [-144.,   0.,  192.]]))

.. seealso:: NLPs can also be created directly from NL-Files using the
             :py:class:`AmplNLP
             <pyomo.contrib.pynumero.interfaces.nlp.AmplNLP>` interface.
