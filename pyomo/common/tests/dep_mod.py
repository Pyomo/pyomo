#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import

__version__ = '1.5'

numpy, numpy_available = attempt_import('numpy', defer_check=True)

bogus_nonexisting_module, bogus_nonexisting_module_available \
    = attempt_import('bogus_nonexisting_module',
                     alt_names=['bogus_nem'],
                     defer_check=True)
