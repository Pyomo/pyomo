{{ name | escape | underline}}

({{ objtype }} from :py:mod:`{{ module }}`)

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

.. auto{{ objtype }}:: {{ objname }}
