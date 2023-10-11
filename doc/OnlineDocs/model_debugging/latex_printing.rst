Latex Printing
==============

Pyomo models can be printed to a LaTeX compatible format using the ``pyomo.util.latex_printer`` function:

.. py:function:: latex_printer(pyomo_component, latex_component_map=None, write_object=None, use_equation_environment=False, split_continuous_sets=False, use_short_descriptors=False, fontsize = None, paper_dimensions=None)

    Prints a pyomo element (Block, Model, Objective, Constraint, or Expression) to a LaTeX compatible string

   :param pyomo_component: The Pyomo component to be printed
   :type pyomo_component: _BlockData or Model or Objective or Constraint or Expression
   :param latex_component_map: A map keyed by Pyomo component, values become the latex representation in the printer
   :type latex_component_map: pyomo.common.collections.component_map.ComponentMap
   :param write_object: The object to print the latex string to.  Can be an open file object, string I/O object, or a string for a filename to write to
   :type write_object: io.TextIOWrapper or io.StringIO or str
   :param use_equation_environment: If False, the equation/aligned construction is used to create a single LaTeX equation.  If True, then the align environment is used in LaTeX and each constraint and objective will be given an individual equation number
   :type use_equation_environment: bool
   :param split_continuous_sets: If False, all sums will be done over 'index in set' or similar.  If True, sums will be done over 'i=1' to 'N' or similar if the set is a continuous set
   :type split_continuous_sets: bool
   :param use_short_descriptors: If False, will print full 'minimize' and 'subject to' etc.  If true, uses 'min' and 's.t.' instead
   :type use_short_descriptors: bool 
   :param fontsize: Sets the font size of the latex output when writing to a file.  Can take in any of the latex font size keywords ['tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large', 'Large', 'LARGE', 'huge', 'Huge'], or an integer referenced off of 'normalsize' (ex: small is -1, Large is +2)
   :type fontsize: str or int
   :param paper_dimensions: A dictionary that controls the paper margins and size.  Keys are: [ 'height', 'width', 'margin_left', 'margin_right', 'margin_top', 'margin_bottom' ].  Default is standard 8.5x11 with one inch margins.  Values are in inches 
   :type paper_dimensions: dict


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

    >>> import pyomo.environ as pe
    >>> from pyomo.util.latex_printer import latex_printer

    >>> m = pe.ConcreteModel(name = 'basicFormulation')
    >>> m.x = pe.Var()
    >>> m.y = pe.Var()
    >>> m.z = pe.Var()
    >>> m.c = pe.Param(initialize=1.0, mutable=True)
    >>> m.objective    = pe.Objective( expr = m.x + m.y + m.z )
    >>> m.constraint_1 = pe.Constraint(expr = m.x**2 + m.y**2.0 - m.z**2.0 <= m.c )

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

A Constraint with a Set
+++++++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pe
    >>> from pyomo.util.latex_printer import latex_printer
    >>> m = pe.ConcreteModel(name='basicFormulation')
    >>> m.I = pe.Set(initialize=[1, 2, 3, 4, 5])
    >>> m.v = pe.Var(m.I)

    >>> def ruleMaker(m): return sum(m.v[i] for i in m.I) <= 0

    >>> m.constraint = pe.Constraint(rule=ruleMaker)

    >>> pstr = latex_printer(m.constraint)

Using a ComponentMap
++++++++++++++++++++

.. doctest::

    >>> import pyomo.environ as pe
    >>> from pyomo.util.latex_printer import latex_printer
    >>> from pyomo.common.collections.component_map import ComponentMap

    >>> m = pe.ConcreteModel(name='basicFormulation')
    >>> m.I = pe.Set(initialize=[1, 2, 3, 4, 5])
    >>> m.v = pe.Var(m.I)

    >>> def ruleMaker(m):  return sum(m.v[i] for i in m.I) <= 0

    >>> m.constraint = pe.Constraint(rule=ruleMaker)

    >>> lcm = ComponentMap()
    >>> lcm[m.v] = 'x'
    >>> lcm[m.I] = ['\\mathcal{A}',['j','k']]

    >>> pstr = latex_printer(m.constraint, latex_component_map=lcm)


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



