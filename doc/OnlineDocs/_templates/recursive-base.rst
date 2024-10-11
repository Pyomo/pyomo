{{ name | escape | underline}}

({{ objtype }} from :py:mod:`{{ module }}`)

.. testsetup:: *

   # import everything from the module containing this class so that
   # doctests for the class docstrings see the correct environment
   from {{ module }} import *

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
