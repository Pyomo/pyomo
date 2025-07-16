{{ name | escape | underline}}

(enum from :py:mod:`{{ module }}`)

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

.. autoenum:: {{ module }}::{{ objname }}
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

   {% block enum_members %}
   {% if enum_members %}
   .. rubric:: {{ _('Enum Members') }}

   {{ member_type }}

   .. autosummary::
      {% for item in enum_members %}
      ~{{ name }}.{{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
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
