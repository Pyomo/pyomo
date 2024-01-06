#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Configuration options for incidence graph generation
"""

import enum
from pyomo.common.config import ConfigDict, ConfigValue, InEnum
from pyomo.common.modeling import NOTSET
from pyomo.repn.plugins.nl_writer import AMPLRepnVisitor, AMPLRepn, text_nl_template
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
    """Use ``pyomo.repn.plugins.nl_writer.AMPLRepnVisitor``"""


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


class _ReconstructVisitor:
    pass


def _amplrepnvisitor_validator(visitor=_ReconstructVisitor):
    # This checks for and returns a valid AMPLRepnVisitor, but I don't want
    # to construct this if we're not using IncidenceMethod.ampl_repn.
    # It is not necessarily the end of the world if we construct this, however,
    # as the code should still work.
    if visitor is _ReconstructVisitor:
        subexpression_cache = {}
        subexpression_order = []
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
            text_nl_template,
            subexpression_cache,
            subexpression_order,
            external_functions,
            var_map,
            used_named_expressions,
            symbolic_solver_labels,
            export_defined_variables,
            sorter,
        )
    elif not isinstance(visitor, AMPLRepnVisitor):
        raise TypeError(
            "'visitor' config argument should be an instance of AMPLRepnVisitor"
        )
    else:
        amplvisitor = visitor
    return amplvisitor


_ampl_repn_visitor = ConfigValue(
    default=_ReconstructVisitor,
    domain=_amplrepnvisitor_validator,
    description="Visitor used to generate AMPLRepn of each constraint",
)


class _IncidenceConfigDict(ConfigDict):
    def __call__(
        self,
        value=NOTSET,
        default=NOTSET,
        domain=NOTSET,
        description=NOTSET,
        doc=NOTSET,
        visibility=NOTSET,
        implicit=NOTSET,
        implicit_domain=NOTSET,
        preserve_implicit=False,
    ):
        init_value = value
        new = super().__call__(
            value=value,
            default=default,
            domain=domain,
            description=description,
            doc=doc,
            visibility=visibility,
            implicit=implicit,
            implicit_domain=implicit_domain,
            preserve_implicit=preserve_implicit,
        )

        if (
            new.method == IncidenceMethod.ampl_repn
            and "ampl_repn_visitor" not in init_value
        ):
            new.ampl_repn_visitor = _ReconstructVisitor

        return new


IncidenceConfig = _IncidenceConfigDict()
"""Options for incidence graph generation

- ``include_fixed`` -- Flag indicating whether fixed variables should be included
  in the incidence graph
- ``linear_only`` -- Flag indicating whether only variables that participate linearly
  should be included.
- ``method`` -- Method used to identify incident variables. Must be a value of the
  ``IncidenceMethod`` enum.
- ``ampl_repn_visitor`` -- Expression visitor used to generate ``AMPLRepn`` of each
  constraint. Must be an instance of ``AMPLRepnVisitor``.

"""


IncidenceConfig.declare("include_fixed", _include_fixed)


IncidenceConfig.declare("linear_only", _linear_only)


IncidenceConfig.declare("method", _method)


IncidenceConfig.declare("ampl_repn_visitor", _ampl_repn_visitor)
