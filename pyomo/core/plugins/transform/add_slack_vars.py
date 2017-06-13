import pyomo.environ
from pyomo.core import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory

from pyomo.util.plugin import alias
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation

from random import randint

#DEBUG
import pdb

class AddSlackVariables(NonIsomorphicTransformation):
    """
    This plugin adds slack variables to every constraint
    """

    alias('core.add_slack_variables', \
          doc="Create a model where we had slack variables to every constraint and add new objective penalizing the sum of the slacks")

    def __init__(self, **kwds):
        kwds['name'] = "add_slack_vars"
        super(AddSlackVariables, self).__init__(**kwds)

    def _get_unique_name(self, instance, name):
        # test if this name already exists in model. If not, we're good. Else, we add random numbers until it doesn't
        # TODO: is this good enough?
        while True:
            if not instance.component(name):
                return name
            else:
                name += str(randint(0,9))

    # JDS: TODOs:
    #  - support targets= argument to specify the constraints to relax
    #    - make sure that works for apply_to and crate_using
    #    - make sure that works for Constraint, IndexedConstraint, and _ConstraintData
    #  - make sure no bogus kwds are passed!

    def _apply_to(self, instance, **kwds):
        # deactivate the objective
        for o in instance.component_data_objects(Objective):
            o.deactivate()

        # create block where we can add slack variables safely
        xblockname = self._get_unique_name(instance, "_core_add_slack_variables")
        instance.add_component(xblockname, Block())
        xblock = instance.component(xblockname)

        obj_expr = 0
        for cons in instance.component_data_objects(Constraint, descend_into=(Block, Disjunct)):
            if (cons.lower is not None and cons.upper is not None) and \
               value(cons.lower) > value(cons.upper):
                # this is a structural infeasibility so slacks aren't going to help:
                raise RuntimeError("Lower bound exceeds upper bound in constraint %s" % cons.name)
            if cons.lower is not None:
                # we add positive slack variable to body:
                # declare positive slack
                varName = "_slack_plus_" + cons.name
                posSlack = Var(within=NonNegativeReals)
                # TODO: Should I be making my own block and adding variables to that?
                # Did we talk about that??
                xblock.add_component(varName, posSlack)
                # add positive slack to body expression
                cons._body += posSlack
                # penalize slack in objective
                obj_expr += posSlack
            if cons.upper is not None:
                # we subtract a positive slack variable from the body:
                # declare slack
                varName = "_slack_minus_" + cons.name
                negSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, negSlack)
                # add negative slack to body expression
                cons._body -= negSlack
                # add slack to objective
                obj_expr += negSlack

        # make a new objective that minimizes sum of slack variables
        xblock._slack_objective = Objective(expr=obj_expr)
