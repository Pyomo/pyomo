#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import PY3 as _PY3
if _PY3:
    from collections.abc import MutableMapping as MutableMapping
    from collections.abc import Mapping as Mapping
    from collections.abc import Sequence as Sequence
else:
    from collections import MutableMapping as MutableMapping
    from collections import Mapping as Mapping
    from collections import Sequence as Sequence

from .orderedset import OrderedDict, OrderedSet
from .component_map import ComponentMap
from .component_set import ComponentSet

from pyutilib.misc import Bunch, Container, Options
