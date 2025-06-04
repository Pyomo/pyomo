Using Solvers with Pyomo
========================

Pyomo supports modeling and scripting but does not install a solver automatically.
For numerous reasons (including stability and managing intermittent dependency
conflicts), Pyomo does not bundle solvers or have strict dependencies on any
third-party solvers. The table below lists solvers that Pyomo supports and
includes guidance on how to install them using ``pip`` or ``conda``.

.. |br| raw:: html

   <br />

.. list-table:: Available Solvers through ``pip`` and ``conda``
   :header-rows: 1

   * - Solver
     - ``pip``
     - ``conda``
     - Documentation
   * - cplex
     - ``pip install cplex``
     - ``conda install -c ibmdecisionoptimization cplex``
     - `Documentation <https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-installing>`__
   * - cyipopt
     - ``pip install cyipopt``
     - ``conda install -c conda-forge cyipopt``
     - `Documentation <https://cyipopt.readthedocs.io/en/stable/install.html>`__
   * - docplex
     - ``pip install docplex``
     - ``conda install -c ibmdecisionoptimization docplex``
     - `Documentation <https://ibmdecisionoptimization.github.io/docplex-doc/getting_started_python.html>`__
   * - glpk
     - N/A
     - ``conda install -c conda-forge glpk``
     - `Documentation <https://www.gnu.org/software/glpk/>`__
   * - Gurobi
     - ``pip install gurobipy``
     - ``conda install -c gurobi gurobi``
     - `Documentation <https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python>`__
   * - HiGHS
     - ``pip install highspy``
     - ``conda install -c conda-forge highspy``
     - `Documentation <https://ergo-code.github.io/HiGHS/dev/interfaces/python/>`__
   * - MAiNGO
     - ``pip install maingopy``
     - N/A
     - `Documentation <https://avt-svt.pages.rwth-aachen.de/public/maingo/install.html#get_maingo>`__
   * - PyMUMPS
     - ``pip install pymumps``
     - ``conda install -c conda-forge pymumps``
     - `Documentation <https://github.com/pymumps/pymumps>`__
   * - SCIP
     - N/A
     - ``conda install -c conda-forge scip``
     - `Documentation <https://www.scipopt.org/index.php#download>`__
   * - XPRESS
     - ``pip install xpress``
     - ``conda install -c fico-xpress xpress``
     - `Documentation <https://www.fico.com/fico-xpress-optimization/docs/dms2019-02/solver/optimizer/python/HTML/chIntro_sec_secInstall.html>`__


.. note::

   We compiled this table of solvers to help you get started, but we encourage
   you to consult the official documentation of your desired solver for the most
   up-to-date and detailed information.