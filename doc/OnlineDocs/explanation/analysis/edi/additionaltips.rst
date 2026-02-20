Additional Tips
---------------

* Developers may need to install the following additional packages:

::

   pip install pytest
   pip install pytest-cov
   pip install sphinx
   pip install sphinx_rtd_theme
   pip install sphinx_copybutton


* If you wish to build the documentation locally, use:

::

   cd <path_to_pyomo>/pyomo/doc/OnlineDocs
   make html

then open the file ``<path_to_pyomo/doc/OnlineDocs/_build/html/index.html>``


* Unit tests and coverage can be run locally using:

::

   cd <path_to_pyomo>/pyomo/pyomo/contrib/edi
   pytest --cov-report term-missing --cov=pyomo.contrib.edi -v ./tests/

or generating html output:

::

   cd <path_to_pyomo>/pyomo/pyomo/contrib/edi
   pytest --cov-report html --cov=pyomo.contrib.edi -v ./tests/
