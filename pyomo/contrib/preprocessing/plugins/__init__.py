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


def load():
    import pyomo.contrib.preprocessing.plugins.deactivate_trivial_constraints
    import pyomo.contrib.preprocessing.plugins.detect_fixed_vars
    import pyomo.contrib.preprocessing.plugins.init_vars
    import pyomo.contrib.preprocessing.plugins.remove_zero_terms
    import pyomo.contrib.preprocessing.plugins.equality_propagate
    import pyomo.contrib.preprocessing.plugins.strip_bounds
    import pyomo.contrib.preprocessing.plugins.zero_sum_propagator
    import pyomo.contrib.preprocessing.plugins.bounds_to_vars
    import pyomo.contrib.preprocessing.plugins.var_aggregator
    import pyomo.contrib.preprocessing.plugins.induced_linearity
    import pyomo.contrib.preprocessing.plugins.constraint_tightener
    import pyomo.contrib.preprocessing.plugins.int_to_binary
