Latex Printing
==============

Pyomo models can be printed to a LaTeX compatible format using the ``pyomo.contrib.latex_printer.latex_printer`` function:

.. autofunction::  pyomo.contrib.latex_printer.latex_printer.latex_printer
   :noindex:

.. note::

    If operating in a Jupyter Notebook, it may be helpful to use:

    ``from IPython.display import display, Math``

    ``display(Math(latex_printer(m))``

Examples
--------

A Model
+++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()
    >>> m.z = pyo.Var()
    >>> m.c = pyo.Param(initialize=1.0, mutable=True)
    >>> m.objective    = pyo.Objective( expr = m.x + m.y + m.z )
    >>> m.constraint_1 = pyo.Constraint(expr = m.x**2 + m.y**2.0 - m.z**2.0 <= m.c )

    >>> pstr = latex_printer(m)


A Constraint
++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()

    >>> m.constraint_1 = pyo.Constraint(expr = m.x**2 + m.y**2 <= 1.0)

    >>> pstr = latex_printer(m.constraint_1)

A Constraint with Set Summation
+++++++++++++++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.latex_printer import latex_printer
    >>> m = pyo.ConcreteModel(name='basicFormulation')
    >>> m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    >>> m.v = pyo.Var(m.I)

    >>> def ruleMaker(m): return sum(m.v[i] for i in m.I) <= 0

    >>> m.constraint = pyo.Constraint(rule=ruleMaker)

    >>> pstr = latex_printer(m.constraint)

Using a ComponentMap to Specify Names
+++++++++++++++++++++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.latex_printer import latex_printer
    >>> from pyomo.common.collections.component_map import ComponentMap

    >>> m = pyo.ConcreteModel(name='basicFormulation')
    >>> m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    >>> m.v = pyo.Var(m.I)

    >>> def ruleMaker(m):  return sum(m.v[i] for i in m.I) <= 0

    >>> m.constraint = pyo.Constraint(rule=ruleMaker)

    >>> lcm = ComponentMap()
    >>> lcm[m.v] = 'x'
    >>> lcm[m.I] = ['\\mathcal{A}',['j','k']]

    >>> pstr = latex_printer(m.constraint, latex_component_map=lcm)


An Expression
+++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()

    >>> m.expression_1 = pyo.Expression(expr = m.x**2 + m.y**2)

    >>> pstr = latex_printer(m.expression_1)


A Simple Expression
+++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()

    >>> pstr = latex_printer(m.x + m.y)



