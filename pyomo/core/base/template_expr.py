#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.template_expr import (
    IndexTemplate, _GetItemIndexer, TemplateExpressionError
)

from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.core.base.template_expr module is deprecated.  '
    'Import expression template objects from pyomo.core.expr.template_expr.',
    version='5.7')
