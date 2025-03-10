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

"This is the deprecated pyomo.core.base.plugin module"

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
