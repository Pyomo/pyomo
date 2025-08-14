#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.opt.base.solvers import LegacySolverFactory
from pyomo.common.factory import Factory
from pyomo.contrib.solver.common.base import LegacySolverWrapper


class SolverFactoryClass(Factory):
    """Factory class for generating instances of solver interfaces (API v2)"""

    def register(self, name, legacy_name=None, doc=None):
        """Register a new solver with this solver factory

        This will register the solver both with this
        :attr:`SolverFactory` and with the original (legacy)
        :attr:`~pyomo.opt.base.solvers.LegacySolverFactory`

        Examples
        --------

        .. testcode::
           :hide:

           SolverFactory = SolverFactoryClass()

        This method can either be called as a decorator on a solver
        interface class definition, e.g.:

        .. testcode::

           from pyomo.contrib.solver.common.base import SolverBase

           @SolverFactory.register("test_solver_1")
           class TestSolver1(SolverBase):
               pass

        Or explicitly:

        .. testcode::

           class TestSolver2(SolverBase):
               pass

           SolverFactory.register("test_solver_2")(TestSolver2)

        When called explicitly, you can pass a custom class to register
        with the :attr:`LegacySolverFactory`:

        .. testcode::

           from pyomo.contrib.solver.common.base import LegacySolverWrapper

           class LegacyTestSolver2(LegacySolverWrapper, TestSolver2):
               pass

           SolverFactory.register("test_solver_2a")(TestSolver2, LegacyTestSolver2)


        Parameters
        ----------
        name : str

            The name used to register this solver interface class

        legacy_name : str

            The name to use to register the legacy interface wrapper to
            this solver interface in the LegacySolverInterface.  If
            ``None``, then ``name`` will be used.

        doc : str

            Extended description of this solver interface.

        """
        if legacy_name is None:
            legacy_name = name

        def decorator(cls, legacy_cls=None):
            self._cls[name] = cls
            self._doc[name] = doc

            if legacy_cls is None:

                class LegacySolver(LegacySolverWrapper, cls):
                    pass

                legacy_cls = LegacySolver

            LegacySolverFactory.register(
                legacy_name, doc + " (new interface)" if doc else doc
            )(legacy_cls)

            # Preserve the preferred name, as registered in the Factory
            cls.name = name
            return cls

        return decorator


#: Global registry/factory for "v2" solver interfaces.
SolverFactory: SolverFactoryClass = SolverFactoryClass()
