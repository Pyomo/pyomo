#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
try:
    import numpy as np
    # Note: sparse.BlockVector leverages the __array__ufunc__ interface
    # released in numpy 1.13
    numpy_available = np.lib.NumpyVersion(np.__version__) >= '1.13.0'
    if not numpy_available:
        import pyomo.common  # ...to set up the logger
        import logging
        logging.getLogger('pyomo.contrib.pynumero').warn(
            "Pynumero requires numpy>=1.13.0; found %s" % (np.__version__,))
except ImportError:
    numpy_available = False

try:
    import scipy
    scipy_available = True
except ImportError:
    scipy_available = False
    import pyomo.common  # ...to set up the logger
    import logging
    logging.getLogger('pyomo.contrib.pynumero').warn(
        "Scipy not available. Install scipy before using pynumero")

if numpy_available:
    from .sparse.intrinsic import *
else:
    # In general, generating output in __init__.py is undesirable, as
    # many __init__.py get imported automatically by pyomo.environ.
    # Fortunately, at the moment, pynumero doesn't implement any
    # plugins, so pyomo.environ ignores it.  When we start implementing
    # general solvers in pynumero we will want to remove / move this
    # warning somewhere deeper in the code.
    import pyomo.common  # ...to set up the logger
    import logging
    logging.getLogger('pyomo.contrib.pynumero').warn(
        "Numpy not available. Install numpy>=1.13.0 before using pynumero")
