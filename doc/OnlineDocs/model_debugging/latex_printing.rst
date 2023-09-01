Latex Printing
==============

Pyomo models can be printed to a LaTeX compatible format using the ``pyomo.util.latex_printer`` function:

.. py:function:: latex_printer(pyomoElement, filename=None, useAlignEnvironment=False, splitContinuousSets=False)

    Prints a pyomo element (Block, Model, Objective, Constraint, or Expression) to a LaTeX compatible string

   :param pyomoElement: The pyomo element to be printed
   :type  pyomoElement: _BlockData or Model or Objective or Constraint or Expression
   :param filename: An optional filename where the latex will be saved
   :type  filename: str
   :param useAlignEnvironment: If False, the equation/aligned construction is used to create a single LaTeX equation.  If True, then the align environment is used in LaTeX and each constraint and objective will be given an individual equation number
   :type useAlignEnvironment: bool
   :param splitContinuousSets: If False, all sums will be done over 'index in set' or similar.  If True, sums will be done over 'i=1' to 'N' or similar if the set is a continuous set
   :type splitContinuousSets: bool


   :return: A LaTeX style string that represents the passed in pyomoElement
   :rtype: str


.. note::

    If operating in a Jupyter Notebook, it may be helpful to use:

    ``from IPython.display import display, Math``

    ``display(Math(latex_printer(m))``


Examples
--------

A Model
+++++++

.. doctest::

    >>> # Note: this model is not mathematically sensible

    >>> import pyomo.environ as pe
    >>> from pyomo.core.expr import Expr_if
    >>> from pyomo.core.base import ExternalFunction
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pe.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pe.Var()
    >>> m.y = pe.Var()
    >>> m.z = pe.Var()
    >>> m.objective    = pe.Objective( expr = m.x + m.y + m.z )
    >>> m.constraint_1 = pe.Constraint(expr = m.x**2 + m.y**-2.0 - m.x*m.y*m.z  + 1 == 2.0)
    >>> m.constraint_2 = pe.Constraint(expr = abs(m.x/m.z**-2) * (m.x + m.y) <= 2.0)
    >>> m.constraint_3 = pe.Constraint(expr = pe.sqrt(m.x/m.z**-2) <= 2.0)
    >>> m.constraint_4 = pe.Constraint(expr = (1,m.x,2))
    >>> m.constraint_5 = pe.Constraint(expr = Expr_if(m.x<=1.0, m.z, m.y) <= 1.0)

    >>> def blackbox(a, b): return sin(a - b)
    >>> m.bb = ExternalFunction(blackbox)
    >>> m.constraint_6 = pe.Constraint(expr= m.x + m.bb(m.x,m.y) == 2 )

    >>> m.I = pe.Set(initialize=[1,2,3,4,5])
    >>> m.J = pe.Set(initialize=[1,2,3])
    >>> m.u = pe.Var(m.I*m.I)
    >>> m.v = pe.Var(m.I)
    >>> m.w = pe.Var(m.J)

    >>> def ruleMaker(m,j): return (m.x + m.y) * sum( m.v[i] + m.u[i,j]**2 for i in m.I ) <= 0
    >>> m.constraint_7 = pe.Constraint(m.I, rule = ruleMaker)

    >>> def ruleMaker(m): return (m.x + m.y) * sum( m.w[j] for j in m.J )
    >>> m.objective_2  = pe.Objective(rule = ruleMaker)

    >>> pstr = latex_printer(m)


A Constraint
++++++++++++

.. doctest::

    >>> import pyomo.environ as pe
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pe.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pe.Var()
    >>> m.y = pe.Var()

    >>> m.constraint_1 = pe.Constraint(expr = m.x**2 + m.y**2 <= 1.0)

    >>> pstr = latex_printer(m.constraint_1)


An Expression
+++++++++++++

.. doctest::

    >>> import pyomo.environ as pe
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pe.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pe.Var()
    >>> m.y = pe.Var()

    >>> m.expression_1 = pe.Expression(expr = m.x**2 + m.y**2)

    >>> pstr = latex_printer(m.expression_1)


A Simple Expression
+++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pe
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pe.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pe.Var()
    >>> m.y = pe.Var()

    >>> pstr = latex_printer(m.x + m.y)



