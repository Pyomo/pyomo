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

from pyomo.common.dependencies import numpy as np, numpy_available


#
# Note: the "if numpy_available" in the class definition also ensures
# that the numpy types are registered if numpy is in fact available
#
class NumericNDArray(np.ndarray if numpy_available else object):
    """An ndarray subclass that stores Pyomo numeric expressions"""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            # Convert all incoming types to ndarray (to prevent recursion)
            args = [np.asarray(i) for i in inputs]
            # Set the return type to be an 'object'.  This prevents the
            # logical operators from casting the result to a bool.  This
            # requires numpy >= 1.6
            kwargs['dtype'] = object

        # Delegate to the base ufunc, but return an instance of this
        # class so that additional operators hit this method.
        ans = getattr(ufunc, method)(*args, **kwargs)
        if isinstance(ans, np.ndarray):
            if ans.size == 1:
                return ans[0]
            return ans.view(NumericNDArray)
        else:
            return ans
