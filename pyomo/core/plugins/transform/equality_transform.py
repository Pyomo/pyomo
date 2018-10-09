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
from pyomo.core.base.misc import create_name

from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.util import collectAbstractComponents


@TransformationFactory.register("core.add_slack_vars", doc="Create an equivalent model by introducing slack variables to eliminate inequality constraints.")
class EqualityTransform(IsomorphicTransformation):
    """
    Creates a new, equivalent model by introducing slack and excess variables
    to eliminate inequality constraints.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.pop("name", "add_slack_vars")
        super(EqualityTransform, self).__init__(**kwds)

    def _create_using(self, model, **kwds):
        """
        Eliminate inequality constraints.

        Required arguments:

          model The model to transform.

        Optional keyword arguments:

          slack_root  The root name of auxiliary slack variables.
                      Default is 'auxiliary_slack'.
          excess_root The root name of auxiliary slack variables.
                      Default is 'auxiliary_excess'.
          lb_suffix   The suffix applied to converted upper bound constraints
                      Default is '_lower_bound'.
          ub_suffix   The suffix applied to converted lower bound constraints
                      Default is '_upper_bound'.
        """

        # Optional naming schemes
        slack_suffix = kwds.pop("slack_suffix", "slack")
        excess_suffix = kwds.pop("excess_suffix", "excess")
        lb_suffix = kwds.pop("lb_suffix", "lb")
        ub_suffix = kwds.pop("ub_suffix", "ub")

        equality = model.clone()
        components = collectAbstractComponents(equality)

        #
        # Fix all Constraint objects
        #
        for con_name in components["Constraint"]:
            con = equality.__getattribute__(con_name)

            #
            # Get all _ConstraintData objects
            #
            # We need to get the keys ahead of time because we are modifying
            # con._data on-the-fly.
            #
            indices = con._data.keys()
            for (ndx, cdata) in [(ndx, con._data[ndx]) for ndx in indices]:

                qualified_con_name = create_name(con_name, ndx)

                # Do nothing with equality constraints
                if cdata.equality:
                    continue

                # Add an excess variable if the lower bound exists
                if cdata.lower is not None:

                    # Make the excess variable
                    excess_name = "%s_%s" % (qualified_con_name, excess_suffix)
                    equality.__setattr__(excess_name,
                                         Var(within=NonNegativeReals))

                    # Make a new lower bound constraint
                    lb_name = "%s_%s" % (create_name("", ndx), lb_suffix)
                    excess = equality.__getattribute__(excess_name)
                    new_expr = (cdata.lower == cdata.body - excess)
                    con.add(lb_name, new_expr)

                # Add a slack variable if the lower bound exists
                if cdata.upper is not None:

                    # Make the excess variable
                    slack_name = "%s_%s" % (qualified_con_name, slack_suffix)
                    equality.__setattr__(slack_name,
                                         Var(within=NonNegativeReals))

                    # Make a new upper bound constraint
                    ub_name = "%s_%s" % (create_name("", ndx), ub_suffix)
                    slack = equality.__getattribute__(slack_name)
                    new_expr = (cdata.upper == cdata.body + slack)
                    con.add(ub_name, new_expr)

                # Since we explicitly `continue` for equality constraints, we
                # can safely remove the old _ConstraintData object
                del con._data[ndx]

        return equality.create()
