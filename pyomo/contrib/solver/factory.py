#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.opt.base.solvers import LegacySolverFactory
from pyomo.common.factory import Factory
from pyomo.contrib.solver.base import LegacySolverWrapper


class SolverFactoryClass(Factory):
    def register(self, name, legacy_name=None, doc=None):
        if legacy_name is None:
            legacy_name = name

        def decorator(cls):
            self._cls[name] = cls
            self._doc[name] = doc

            class LegacySolver(LegacySolverWrapper, cls):
                pass

            LegacySolverFactory.register(legacy_name, doc + " (new interface)")(
                LegacySolver
            )

            # Preserve the preferred name, as registered in the Factory
            cls.name = name
            return cls

        return decorator


SolverFactory = SolverFactoryClass()
