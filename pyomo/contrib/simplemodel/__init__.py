try:
    from pyomocontrib_simplemodel import *
except:
    # Only raise exception if nose is NOT running
    import sys
    if 'nose' not in sys.modules and 'nose2' not in sys.modules:
        raise RuntimeError(
            "The pyomocontrib_simplemodel package is not installed.")
