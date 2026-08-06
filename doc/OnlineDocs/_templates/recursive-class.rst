{{ name | escape | underline}}

(class from :py:mod:`{{ module }}`)

.. testsetup:: *

   # import everything from the module containing this class so that
   # doctests for the class docstrings see the correct environment
   from {{ module }} import *
   try:
       from {{ module }} import _autosummary_doctest_setup
       _autosummary_doctest_setup()
   except ImportError:
       pass

.. currentmodule:: {{ module }}

{# Note that numpy.ndarray examples fail doctest; disable documentation
   of inherited members for classes derived from ndarray #}

.. autoclass:: {{ module }}::{{ objname }}
   :members:
   :show-inheritance:
   {{ '' if (module + '.' + name) in (
         'pyomo.contrib.pynumero.sparse.block_vector.BlockVector',
         'pyomo.contrib.pynumero.sparse.mpi_block_vector.MPIBlockVector',
         'pyomo.core.expr.ndarray.NumericNDArray',
      ) else ':inherited-members:' }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. rubric:: Member Documentation
