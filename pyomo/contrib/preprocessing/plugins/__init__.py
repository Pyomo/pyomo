def load():
    import pyomo.contrib.preprocessing.plugins.deactivate_trivial_constraints
    import pyomo.contrib.preprocessing.plugins.detect_fixed_vars
    import pyomo.contrib.preprocessing.plugins.init_vars
    import pyomo.contrib.preprocessing.plugins.remove_zero_terms
    import pyomo.contrib.preprocessing.plugins.equality_propagate
    import pyomo.contrib.preprocessing.plugins.strip_bounds
