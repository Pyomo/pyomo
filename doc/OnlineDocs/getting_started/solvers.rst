.. -*- mode: rst -*-

Using Solvers with Pyomo
========================

Pyomo supports modeling and scripting but does not install a solver automatically.
For numerous reasons (including stability and managing intermittent dependency
conflicts), Pyomo does not bundle solvers or have strict dependencies on any
third-party solvers. The table below lists a subset of the solvers compatible with Pyomo that can be installed using ``pip`` or ``conda``. It includes
both commercial and open-source solvers -- users are responsible for understanding
the license requirements for their desired solver.

.. |br| raw:: html

   <br />

..
    NOTE the use of Unicode nonbreaking spaces (xA0) and hyphens (x2011)
    in the PIP and CONDA command lines so that the commands render
    sensibly

.. list-table:: Available Solvers through ``pip`` and ``conda``
   :header-rows: 1

   * - Solver
     - Pip
     - Conda
     - License |br| Docs
   * - cplex
     - ``pip install cplex``
     - ``conda install ‑c ibmdecisionoptimization \    cplex``
     - `License <https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing>`__
       `Docs <https://www.ibm.com/docs/en/icos/latest?topic=cplex-installing>`__
   * - CPoptimizer
     - ``pip install cplex \    docplex``
     - ``conda install ‑c ibmdecisionoptimization \    cplex docplex``
     - `License <https://github.com/IBMDecisionOptimization/docplex-doc/blob/master/LICENSE.txt>`__
       `Docs <https://ibmdecisionoptimization.github.io/docplex-doc/getting_started_python.html>`__
   * - cyipopt
     - ``pip install cyipopt``
     - ``conda install ‑c conda‑forge cyipopt``
     - `License <https://cyipopt.readthedocs.io/en/stable/#copyright>`__
       `Docs <https://cyipopt.readthedocs.io/en/stable/install.html>`__
   * - glpk
     - N/A
     - ``conda install ‑c conda‑forge glpk``
     - `License <https://www.gnu.org/licenses/licenses.html>`__
       `Docs <https://www.gnu.org/software/glpk/>`__
   * - Gurobi
     - ``pip install gurobipy``
     - ``conda install ‑c gurobi gurobi``
     - `License <https://www.gurobi.com/solutions/licensing/>`__
       `Docs <https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python>`__
   * - HiGHS
     - ``pip install highspy``
     - ``conda install ‑c conda‑forge highspy``
     - `License <https://github.com/ERGO-Code/HiGHS/blob/master/LICENSE.txt>`__
       `Docs <https://ergo-code.github.io/HiGHS/dev/interfaces/python/>`__
   * - KNITRO
     - ``pip install knitro``
     - N/A
     - `License <https://www.artelys.com/solvers/knitro/>`__
       `Docs <https://www.artelys.com/app/docs/knitro/index.html>`__
   * - MAiNGO
     - ``pip install maingopy``
     - N/A
     - `License <https://git.rwth-aachen.de/avt-svt/public/maingo/-/blob/master/LICENSE>`__
       `Docs <https://avt-svt.pages.rwth-aachen.de/public/maingo/install.html>`__
   * - PyMUMPS
     - ``pip install pymumps``
     - ``conda install ‑c conda‑forge pymumps``
     - `License <https://github.com/PyMumps/pymumps/blob/master/COPYING>`__
       `Docs <https://github.com/pymumps/pymumps>`__
   * - SCIP
     - N/A
     - ``conda install ‑c conda‑forge scip``
     - `License <https://www.scipopt.org/scip/doc/html/LICENSE.php>`__
       `Docs <https://www.scipopt.org/index.php#download>`__
   * - XPRESS
     - ``pip install xpress``
     - ``conda install ‑c fico‑xpress xpress``
     - `License <https://www.fico.com/en/fico-xpress-trial-and-licensing-options>`__
       `Docs <https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/chIntro.html?scroll=secInstall>`__

.. note::

   We compiled this table of solvers to help you get started, but we encourage
   you to consult the official documentation of your desired solver for the most
   up-to-date and detailed information.
