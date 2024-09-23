Gurobi
======


Handling Gurobi licenses through the APPSI interface
----------------------------------------------------

In order to obtain performance benefits when re-solving a Pyomo model
with Gurobi repeatedly, Pyomo has to keep a reference to a gurobipy
model between calls to
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.solve()`. Depending
on the Gurobi license type, this may "consume" a license as long as
any APPSI-Gurobi interface exists (i.e., has not been garbage
collected). To release a Gurobi license for other processes, use the
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.release_license()`
method as shown below. Note that
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.release_license()`
must be called on every instance for this to actually release the
license. However, releasing the license will delete the gurobipy model
which will have to be reconstructed from scratch the next time
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.solve()` is
called, negating any performance benefit of the persistent solver
interface.

.. code-block:: python

    >>> opt = appsi.solvers.Gurobi()  # doctest: +SKIP
    >>> results = opt.solve(model)  # doctest: +SKIP
    >>> opt.release_license()  # doctest: +SKIP


Also note that both the
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.available()` and
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.solve()` methods
will construct a gurobipy model, thereby (depending on the type of
license) "consuming" a license. The
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.available()`
method has to do this so that the availability does not change between
calls to
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.available()` and
:py:meth:`~pyomo.contrib.appsi.solvers.gurobi.Gurobi.solve()`, leading
to unexpected errors.


.. autosummary::

   pyomo.contrib.appsi.solvers.gurobi.GurobiResults
   pyomo.contrib.appsi.solvers.gurobi.Gurobi
