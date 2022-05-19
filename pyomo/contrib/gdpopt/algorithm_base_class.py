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

from pyomo.common.config import ConfigValue

class _GDPoptAlgorithm():
    def __init__(self, parent, kwds=None):
        self.parent = parent
        # Transfer the parent config info: we create it if it is not there, and
        # overwrite the values if it is already there. The parent defers to what
        # is in this class during solve. We bypass the parent's getter because
        # that's an infinite loop: we know it doesn't have an impl, we're
        # *setting* it's impl.
        for kwd, val in self.parent._CONFIG.items():
            if kwd not in self.CONFIG:
                self.CONFIG.declare(kwd, ConfigValue(default=val))
            else:
                self.CONFIG[kwd] = val
        # kwds might be a superset of what we saw in the previous loop
        if kwds is not None:
            for kwd, val in kwds.items():
                if kwd not in self.CONFIG and kwd in self.parent._CONFIG:
                    self.CONFIG.declare(kwd, ConfigValue(default=val))
                else:
                    self.CONFIG[kwd] = val

    def _solve_gdp(self, original_model, config):
        # To be implemented by algorithms
        pass
