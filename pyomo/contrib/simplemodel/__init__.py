from pyomo.common.deprecation import deprecation_warning

try:
    deprecation_warning(
        "The use of pyomo.contrib.simple model is deprecated. "
        "This capability is now supported in the pyomo_simplemodel "
        "package, which is included in the pyomo_community distribution.",
        version='5.6.9')
    from pyomocontrib_simplemodel import *
except:
    # Only raise exception if nose is NOT running
    import sys
    if 'nose' not in sys.modules and 'nose2' not in sys.modules:
        raise RuntimeError(
            "The pyomocontrib_simplemodel package is not installed.")
