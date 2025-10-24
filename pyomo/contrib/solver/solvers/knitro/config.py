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

from pyomo.common.config import Bool, ConfigValue
from pyomo.contrib.solver.common.config import SolverConfig


class KnitroConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ) -> None:
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.rebuild_model_on_remove_var: bool = self.declare(
            "rebuild_model_on_remove_var",
            ConfigValue(
                domain=Bool,
                default=False,
                doc=(
                    "KNITRO solver does not allow variable removal. We can "
                    "either make the variable a continuous free variable or "
                    "rebuild the whole model when variable removal is "
                    "attempted. When `rebuild_model_on_remove_var` is set to "
                    "True, the model will be rebuilt."
                ),
            ),
        )

        self.restore_variable_values_after_solve: bool = self.declare(
            "restore_variable_values_after_solve",
            ConfigValue(
                domain=Bool,
                default=False,
                doc=(
                    "To evaluate non-linear constraints, KNITRO solver sets "
                    "explicit values on variables. This option controls "
                    "whether to restore the original variable values after "
                    "solving."
                ),
            ),
        )
