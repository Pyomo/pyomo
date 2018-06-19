"""This module contains functions to interrogate the size of a Pyomo model."""
import logging
from pyomo.core import Constraint
from pyutilib.misc import Bunch


default_logger = logging.getLogger('pyomo.util.model_size')
default_logger.setLevel(logging.INFO)


class ModelSizeReport(Bunch):
    """Stores model size information.

    Active blocks are those who have an active flag of True and whose parent,
    if exists, is an active block or an active Disjunct.

    Active constraints are those with an active flag of True and are reachable
    via an active Block, are on an active Disjunct, or are on a disjunct with
    indicator_var fixed to 1 with active flag True.

    Active variables refer to the presence of the variable on an active
    constraint, or that the variable is an unfixed indicator_var for a Disjunct
    participating in an active Disjunction.

    Active disjuncts refer to disjuncts with an active flag of True and who
    participate in an active Disjunction.

    Active disjunctions follow the same rules as active constraints.

    """
    pass


def build_model_size_report(model):
    """Build a model size report object."""
    report = ModelSizeReport()
    for constr in model.component_data_objects(Constraint):
        pass

    return report
