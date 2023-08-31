#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.opt.base import SolverFactory as LegacySolverFactory
from pyomo.common.factory import Factory
from pyomo.solver.base import LegacySolverInterface


class SolverFactoryClass(Factory):
    def register(self, name, doc=None):
        def decorator(cls):
            self._cls[name] = cls
            self._doc[name] = doc

            class LegacySolver(LegacySolverInterface, cls):
                pass

            LegacySolverFactory.register(name, doc)(LegacySolver)

            return cls

        return decorator


SolverFactory = SolverFactoryClass()
