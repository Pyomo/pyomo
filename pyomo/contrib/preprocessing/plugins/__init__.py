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
    from pyomo.contrib.preprocessing.plugins import (
        deactivate_trivial_constraints,
        detect_fixed_vars,
        init_vars,
        remove_zero_terms,
        equality_propagate,
        strip_bounds,
        zero_sum_propagator,
        bounds_to_vars,
        var_aggregator,
        induced_linearity,
        constraint_tightener,
        int_to_binary,
    )
