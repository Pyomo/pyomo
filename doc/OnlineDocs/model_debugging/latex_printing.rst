Latex Printing
==============

Pyomo models can be printed to a LaTeX compatible format using the ``pyomo.util.latex_printer`` function:

.. py:function:: latex_printer(pyomo_component, latex_component_map=None, write_object=None, use_equation_environment=False, explicit_set_summation=False, use_short_descriptors=False, fontsize = None, paper_dimensions=None)

    Prints a pyomo component (Block, Model, Objective, Constraint, or Expression) to a LaTeX compatible string

   :param pyomo_component: The Pyomo component to be printed
   :type pyomo_component: _BlockData or Model or Objective or Constraint or Expression
   :param latex_component_map: A map keyed by Pyomo component, values become the latex representation in the printer
   :type latex_component_map: pyomo.common.collections.component_map.ComponentMap
   :param write_object: The object to print the latex string to.  Can be an open file object, string I/O object, or a string for a filename to write to
   :type write_object: io.TextIOWrapper or io.StringIO or str
   :param use_equation_environment: LaTeX can render as either a single equation object or as an aligned environment, that in essence treats each objective and constraint as individual numbered equations.  If False, then the align environment is used in LaTeX and each constraint and objective will be given an individual equation number.  If True, the equation/aligned construction is used to create a single LaTeX equation for the entire model.  The align environment (ie, flag==False which is the default) is preferred because it allows for page breaks in large models.
   :type use_equation_environment: bool
   :param explicit_set_summation: If False, all sums will be done over 'index in set' or similar.  If True, sums that have a contiguous set (ex: [1,2,3,4,5...]) will be done over 'i=1' to 'N' or similar 
   :type explicit_set_summation: bool
   :param throw_templatization_error: Option to throw an error on templatization failure rather than printing each constraint individually, useful for very large models
   :type throw_templatization_error: bool


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

    >>> import pyomo.environ as pyo
    >>> from pyomo.util.latex_printer import latex_printer

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
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()

    >>> m.constraint_1 = pyo.Constraint(expr = m.x**2 + m.y**2 <= 1.0)

    >>> pstr = latex_printer(m.constraint_1)

A Constraint with Set Summation
+++++++++++++++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.util.latex_printer import latex_printer
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
    >>> from pyomo.util.latex_printer import latex_printer
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
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()

    >>> m.expression_1 = pyo.Expression(expr = m.x**2 + m.y**2)

    >>> pstr = latex_printer(m.expression_1)


A Simple Expression
+++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pyo
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pyo.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()

    >>> pstr = latex_printer(m.x + m.y)



