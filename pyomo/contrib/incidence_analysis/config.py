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
"""Configuration options for incidence graph generation"""

import enum
from pyomo.common.config import ConfigDict, ConfigValue, InEnum
from pyomo.common.modeling import NOTSET
from pyomo.repn.ampl import AMPLRepnVisitor
from pyomo.repn.util import FileDeterminism, FileDeterminism_to_SortComponents


class IncidenceMethod(enum.Enum):
    """Methods for identifying variables that participate in expressions"""

    identify_variables = 0
    """Use ``pyomo.core.expr.visitor.identify_variables``"""

    standard_repn = 1
    """Use ``pyomo.repn.standard_repn.generate_standard_repn``"""

    standard_repn_compute_values = 2
    """Use ``pyomo.repn.standard_repn.generate_standard_repn`` with
    ``compute_values=True``
    """

    ampl_repn = 3
    """Use ``pyomo.repn.ampl.AMPLRepnVisitor``"""


class IncidenceOrder(enum.Enum):

    dulmage_mendelsohn_upper = 0

    dulmage_mendelsohn_lower = 1


_include_fixed = ConfigValue(
    default=False,
    domain=bool,
    description="Include fixed variables",
    doc=(
        "Flag indicating whether fixed variables should be included in the"
        " incidence graph"
    ),
)


_linear_only = ConfigValue(
    default=False,
    domain=bool,
    description="Identify only variables that participate linearly",
    doc=(
        "Flag indicating whether only variables that participate linearly should"
        " be included."
    ),
)


_method = ConfigValue(
    default=IncidenceMethod.standard_repn,
    domain=InEnum(IncidenceMethod),
    description="Method used to identify incident variables",
)


def _amplrepnvisitor_validator(visitor):
    if not isinstance(visitor, AMPLRepnVisitor):
        raise TypeError(
            "'visitor' config argument should be an instance of AMPLRepnVisitor"
        )
    return visitor


_ampl_repn_visitor = ConfigValue(
    default=None,
    domain=_amplrepnvisitor_validator,
    description="Visitor used to generate AMPLRepn of each constraint",
)


IncidenceConfig = ConfigDict()
"""Options for incidence graph generation

- ``include_fixed`` -- Flag indicating whether fixed variables should be included
  in the incidence graph
- ``linear_only`` -- Flag indicating whether only variables that participate linearly
  should be included.
- ``method`` -- Method used to identify incident variables. Must be a value of the
  ``IncidenceMethod`` enum.
- ``_ampl_repn_visitor`` -- Expression visitor used to generate ``AMPLRepn`` of each
  constraint. Must be an instance of ``AMPLRepnVisitor``. *This option is constructed
  automatically when needed and should not be set by users!*

"""


IncidenceConfig.declare("include_fixed", _include_fixed)


IncidenceConfig.declare("linear_only", _linear_only)


IncidenceConfig.declare("method", _method)


IncidenceConfig.declare("_ampl_repn_visitor", _ampl_repn_visitor)


def get_config_from_kwds(**kwds):
    """Get an instance of IncidenceConfig from provided keyword arguments.

    If the ``method`` argument is ``IncidenceMethod.ampl_repn`` and no
    ``AMPLRepnVisitor`` has been provided, a new ``AMPLRepnVisitor`` is
    constructed. This function should generally be used by callers such
    as ``IncidenceGraphInterface`` to ensure that a visitor is created then
    re-used when calling ``get_incident_variables`` in a loop.

    """
    if (
        kwds.get("method", None) is IncidenceMethod.ampl_repn
        and kwds.get("_ampl_repn_visitor", None) is None
    ):
        subexpression_cache = {}
        external_functions = {}
        var_map = {}
        used_named_expressions = set()
        symbolic_solver_labels = False
        # TODO: Explore potential performance benefit of exporting defined variables.
        # This likely only shows up if we can preserve the subexpression cache across
        # multiple constraint expressions.
        export_defined_variables = False
        sorter = FileDeterminism_to_SortComponents(FileDeterminism.ORDERED)
        amplvisitor = AMPLRepnVisitor(
            subexpression_cache,
            external_functions,
            var_map,
            used_named_expressions,
            symbolic_solver_labels,
            export_defined_variables,
            sorter,
        )
        kwds["_ampl_repn_visitor"] = amplvisitor
    return IncidenceConfig(kwds)
