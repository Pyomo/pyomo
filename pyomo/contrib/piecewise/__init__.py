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

from pyomo.contrib.piecewise.piecewise_linear_expression import (
    PiecewiseLinearExpression,
)
from pyomo.contrib.piecewise.piecewise_linear_function import PiecewiseLinearFunction

## register transformations
from pyomo.contrib.piecewise.transform.inner_representation_gdp import (
    InnerRepresentationGDPTransformation,
)
from pyomo.contrib.piecewise.transform.disaggregated_convex_combination import (
    DisaggregatedConvexCombinationTransformation,
)
from pyomo.contrib.piecewise.transform.outer_representation_gdp import (
    OuterRepresentationGDPTransformation,
)
from pyomo.contrib.piecewise.transform.multiple_choice import (
    MultipleChoiceTransformation,
)
from pyomo.contrib.piecewise.transform.reduced_inner_representation_gdp import (
    ReducedInnerRepresentationGDPTransformation,
)
from pyomo.contrib.piecewise.transform.convex_combination import (
    ConvexCombinationTransformation,
)
from pyomo.contrib.piecewise.transform.nonlinear_to_pwl import (
    DomainPartitioningMethod,
    NonlinearToPWL,
)
from pyomo.contrib.piecewise.transform.nested_inner_repn import (
    NestedInnerRepresentationGDPTransformation,
)
from pyomo.contrib.piecewise.transform.disaggregated_logarithmic import (
    DisaggregatedLogarithmicMIPTransformation,
)
from pyomo.contrib.piecewise.transform.incremental import IncrementalMIPTransformation
from pyomo.contrib.piecewise.triangulations import Triangulation
