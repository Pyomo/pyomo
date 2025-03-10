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


def load():
    from pyomo.gdp.plugins import (
        bigm,
        hull,
        bilinear,
        gdp_var_mover,
        cuttingplane,
        fix_disjuncts,
        partition_disjuncts,
        between_steps,
        multiple_bigm,
        transform_current_disjunctive_state,
        bound_pretransformation,
        binary_multiplication,
    )


#
# declare deprecation paths for removed modules
#
from pyomo.common.deprecation import moved_module

moved_module(
    'pyomo.gdp.plugins.chull',
    'pyomo._archive.chull',
    msg='The pyomo.gdp.plugins.chull module is deprecated.  '
    'Import the Hull reformulation objects from pyomo.gdp.plugins.hull.',
    version='5.7',
)
del moved_module
