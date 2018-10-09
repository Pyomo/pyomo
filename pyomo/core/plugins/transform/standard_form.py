#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import *
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.nonnegative_transform import *
from pyomo.core.plugins.transform.equality_transform import *


@TransformationFactory.register("core.standard_form", doc="Create an equivalent LP model in standard form.")
class StandardForm(IsomorphicTransformation):
    """
    Produces a standard-form representation of the model. This form has 
    the coefficient matrix (A), the cost vector (c), and the
    constraint vector (b), where the 'standard form' problem is

    min/max c'x
    s.t.    Ax = b
            x >= 0

    Options
        slack_names         Default auxiliary_slack
        excess_names        Default auxiliary_excess
        lb_names            Default _lower_bound
        up_names            Default _upper_bound
        pos_suffix          Default _plus
        neg_suffix          Default _neg
    """

    def __init__(self, **kwds):
        kwds['name'] = "standard_form"
        super(StandardForm, self).__init__(**kwds)

    def _create_using(self, model, **kwds):
        """
        Tranform a model to standard form
        """

        # Optional naming schemes to pass to EqualityTransform
        eq_kwds = {}
        eq_kwds["slack_names"] = kwds.pop("slack_names", "auxiliary_slack")
        eq_kwds["excess_names"] = kwds.pop("excess_names", "auxiliary_excess")
        eq_kwds["lb_names"] = kwds.pop("lb_names", "_lower_bound")
        eq_kwds["ub_names"] = kwds.pop("ub_names", "_upper_bound")

        # Optional naming schemes to pass to NonNegativeTransformation
        nn_kwds = {}
        nn_kwds["pos_suffix"] = kwds.pop("pos_suffix", "_plus")
        nn_kwds["neg_suffix"] = kwds.pop("neg_suffix", "_minus")

        nonneg = NonNegativeTransformation()
        equality = EqualityTransform()

        # Since NonNegativeTransform introduces new constraints
        # (that aren't equality constraints) we call it first.
        #
        # EqualityTransform introduces new variables, but they are
        # constrainted to be nonnegative.
        sf = nonneg(model, **nn_kwds)
        sf = equality(sf, **eq_kwds)

        return sf
