Additional Tips
---------------

* If you wish to build the documentation locally, use:

::

   cd <path_to_pyomo>/pyomo/doc/OnlineDocs
   make html

then open the file ``<path_to_pyomo/doc/OnlineDocs/_build/html/index.html>``


* Unit tests and coverage can be run locally using:

::

   cd <path_to_pyomo>/pyomo/pyomo/contrib/edi
   pytest --cov-report term-missing --cov=pyomo.contrib.edi -v ./tests/