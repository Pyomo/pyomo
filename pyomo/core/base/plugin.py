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
import inspect
from pyomo.common.deprecation import deprecation_warning

deprecation_warning(
    "The pyomo.core.base.plugin module is deprecated.  "
    "See pyomo.core.base.transformation for Transformation and "
    "TransformationFactory, pyomo.core.base.component for "
    "ModelComponentFactory and pyomo.scripting.interface for "
    "IPyomoScript* interfaces.",
    version='6.0',
    calling_frame=inspect.currentframe().f_back,
)

__all__ = [
    'pyomo_callback',
    'IPyomoExpression',
    'ExpressionFactory',
    'ExpressionRegistration',
    'IPyomoPresolver',
    'IPyomoPresolveAction',
    'IParamRepresentation',
    'ParamRepresentationFactory',
    'IPyomoScriptPreprocess',
    'IPyomoScriptCreateModel',
    'IPyomoScriptCreateDataPortal',
    'IPyomoScriptModifyInstance',
    'IPyomoScriptPrintModel',
    'IPyomoScriptPrintInstance',
    'IPyomoScriptSaveInstance',
    'IPyomoScriptPrintResults',
    'IPyomoScriptSaveResults',
    'IPyomoScriptPostprocess',
    'ModelComponentFactory',
    'Transformation',
    'TransformationFactory',
]

from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.transformation import (
    Transformation,
    TransformationFactory,
    TransformationData,
    TransformationInfo,
    TransformationTimer,
)

from pyomo.scripting.interface import (
    implements,
    Interface,
    Plugin,
    ExtensionPoint,
    DeprecatedInterface,
    pyomo_callback,
    IPyomoPresolver,
    IPyomoPresolveAction,
    IPyomoScriptPreprocess,
    IPyomoScriptCreateModel,
    IPyomoScriptCreateDataPortal,
    IPyomoScriptModifyInstance,
    IPyomoScriptPrintModel,
    IPyomoScriptPrintInstance,
    IPyomoScriptSaveInstance,
    IPyomoScriptPrintResults,
    IPyomoScriptSaveResults,
    IPyomoScriptPostprocess,
)


class IPyomoExpression(DeprecatedInterface):
    def type(self):
        """Return the type of expression"""

    def create(self, args):
        """Create an instance of this expression type"""


class IParamRepresentation(DeprecatedInterface):
    pass
