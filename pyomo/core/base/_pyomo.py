#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import iteritems
from pyomo.core.base.plugin import (unique_component_name, Factory, implements,
                                    Interface, Plugin, CreatePluginFactory,
                                    ExtensionPoint, TransformationTimer,
                                    registered_callback, pyomo_callback,
                                    IPyomoExpression, ExpressionFactory,
                                    ExpressionRegistration, IPyomoPresolver,
                                    IPyomoPresolveAction, IParamRepresentation,
                                    ParamRepresentationFactory,
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
                                    ModelComponentFactory, Transformation,
                                    TransformationFactory,
                                    ModelComponentFactoryClass,
                                    TransformationInfo, TransformationData,
                                    apply_transformation)

def predefined_sets():
    from pyomo.core.base.set import GlobalSets
    return list((name, obj.doc) for name,obj in iteritems(GlobalSets))


def model_components():
    return [ (name, ModelComponentFactory.doc(name))
             for name in ModelComponentFactory ]
