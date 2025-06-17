{% if fullname == 'pyomo' %}
Library Reference
=================
{% else %}
{{ name | escape | underline}}
{% endif %}

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

.. automodule:: {{ fullname }}
   :undoc-members:

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :toctree:
      :template: recursive-base.rst
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {% block enums %}
   {%- if enums %}
   .. rubric:: {{ _('Enums') }}

   .. autosummary::
      :toctree:
      :template: recursive-enum.rst
   {% for item in enums %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: recursive-class.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
      :template: recursive-class.rst
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :template: recursive-base.rst
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: recursive-module.rst
   :recursive:
{% for item in modules %}
{# Need item != tests for Sphinx >= 8.0; !endswith(.tests) for < 8.0 #}
{% if item != 'tests' and not item.endswith('.tests')
   and item != 'examples' and not item.endswith('.examples') %}
   {{ item }}
{% endif %}
{%- endfor %}
{% endif %}
{%- endblock %}
