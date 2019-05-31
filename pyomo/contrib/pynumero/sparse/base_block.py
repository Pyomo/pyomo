#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import abc
import six

# These classes are for checking types consistently

@six.add_metaclass(abc.ABCMeta)
class BaseBlockVector(object):
    """Base class for block vectors"""

    def __init__(self):
        pass

@six.add_metaclass(abc.ABCMeta)
class BaseBlockMatrix(object):
    """Base class for block matrices"""

    def __init__(self):
        pass
