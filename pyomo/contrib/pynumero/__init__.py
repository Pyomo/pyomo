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
    numpy_available = True
except ImportError:
    numpy_available = False

if numpy_available:
    from .sparse.intrinsic import *
else:
    # In general, generating output in __init__.py is undesirable, as
    # many __init__.py get imported automatically by pyomo.environ.
    # Fortunately, at the moment, pynumero doesn't implement any
    # plugins, so pyomo.environ ignores it.  When we start implementing
    # general solvers in pynumero we will want to remove / move this
    # warning somewhere deeper in the code.
    import logging
    logging.getLogger('pyomo.contrib.pynumero').warn(
        "Numpy not available. Install numpy before using pynumero")





