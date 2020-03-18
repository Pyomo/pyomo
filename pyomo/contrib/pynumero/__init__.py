#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import, scipy, scipy_available

# Note: sparse.BlockVector leverages the __array__ufunc__ interface
# released in numpy 1.13
numpy, numpy_available = attempt_import('numpy', 'Pynumero requires numpy',
                                        minimum_version='1.13.0',
                                        defer_check=False)

if not scipy_available:
    # In general, generating output in __init__.py is undesirable, as
    # many __init__.py get imported automatically by pyomo.environ.
    # Fortunately, at the moment, pynumero doesn't implement any
    # plugins, so pyomo.environ ignores it.  When we start implementing
    # general solvers in pynumero we will want to remove / move this
    # warning somewhere deeper in the code.
    scipy.generate_import_warning('pyomo.contrib.pynumero')

if not numpy_available:
    numpy.generate_import_warning('pyomo.contrib.pynumero')
else:
    from .sparse.intrinsic import *
