#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math
import copy
import re
import pyomo.environ as pyo
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
    Expr_ifExpression,
    ExternalFunctionExpression,
)

from pyomo.core.expr.visitor import identify_components
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
    GetItemExpression,
    GetAttrExpression,
    TemplateSumExpression,
    IndexTemplate,
    Numeric_GetItemExpression,
    templatize_constraint,
    resolve_template,
    templatize_rule,
)
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.param import _ParamData, ScalarParam, IndexedParam
from pyomo.core.base.set import _SetData
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.collections.component_set import ComponentSet

from pyomo.core.base.external import _PythonCallbackFunctionID

from pyomo.core.base.block import _BlockData

from pyomo.repn.util import ExprType

from pyomo.common import DeveloperError

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL


def applySmartVariables(name):
    splitName = name.split('_')
    # print(splitName)

    filteredName = []

    prfx = ''
    psfx = ''
    for i in range(0, len(splitName)):
        se = splitName[i]
        if se != 0:
            if se == 'dot':
                prfx = '\\dot{'
                psfx = '}'
            elif se == 'hat':
                prfx = '\\hat{'
                psfx = '}'
            elif se == 'bar':
                prfx = '\\bar{'
                psfx = '}'
            elif se == 'mathcal':
                prfx = '\\mathcal{'
                psfx = '}'
            else:
                filteredName.append(se)
        else:
            filteredName.append(se)

    joinedName = prfx + filteredName[0] + psfx
    # print(joinedName)
    # print(filteredName)
    for i in range(1, len(filteredName)):
        joinedName += '_{' + filteredName[i]

    joinedName += '}' * (len(filteredName) - 1)
    # print(joinedName)

    return joinedName


# def multiple_replace(pstr, rep_dict):
#     pattern = re.compile("|".join(rep_dict.keys()), flags=re.DOTALL)
#     return pattern.sub(lambda x: rep_dict[x.group(0)], pstr)


def latex_component_map_generator(
    pyomo_component,
    use_smart_variables=False,
    x_only_mode=0,
    overwrite_dict=None,
    # latex_component_map=None,
):
    """This function produces a string that can be rendered as LaTeX

    As described, this function produces a string that can be rendered as LaTeX

    Parameters
    ----------
    pyomo_component: _BlockData or Model or Constraint or Expression or Objective
        The thing to be printed to LaTeX.  Accepts Blocks (including models), Constraints, and Expressions

    filename: str
        An optional file to write the LaTeX to.  Default of None produces no file

    use_equation_environment: bool
        Default behavior uses equation/aligned and produces a single LaTeX Equation (ie, ==False).
        Setting this input to True will instead use the align environment, and produce equation numbers for each
        objective and constraint.  Each objective and constraint will be labeled with its name in the pyomo model.
        This flag is only relevant for Models and Blocks.

    splitContinuous: bool
        Default behavior has all sum indices be over "i \\in I" or similar.  Setting this flag to
        True makes the sums go from: \\sum_{i=1}^{5} if the set I is continuous and has 5 elements

    Returns
    -------
    str
        A LaTeX string of the pyomo_component

    """

    # Various setup things

    # is Single implies Objective, constraint, or expression
    # these objects require a slight modification of behavior
    # isSingle==False means a model or block

    if overwrite_dict is None:
        overwrite_dict = ComponentMap()

    isSingle = False

    if isinstance(
        pyomo_component,
        (pyo.Objective, pyo.Constraint, pyo.Expression, ExpressionBase, pyo.Var),
    ):
        isSingle = True
    elif isinstance(pyomo_component, _BlockData):
        # is not single, leave alone
        pass
    else:
        raise ValueError(
            "Invalid type %s passed into the latex printer"
            % (str(type(pyomo_component)))
        )

    if isSingle:
        temp_comp, temp_indexes = templatize_fcn(pyomo_component)
        variableList = []
        for v in identify_components(
            temp_comp, [ScalarVar, _GeneralVarData, IndexedVar]
        ):
            if isinstance(v, _GeneralVarData):
                v_write = v.parent_component()
                if v_write not in ComponentSet(variableList):
                    variableList.append(v_write)
            else:
                if v not in ComponentSet(variableList):
                    variableList.append(v)

        parameterList = []
        for p in identify_components(
            temp_comp, [ScalarParam, _ParamData, IndexedParam]
        ):
            if isinstance(p, _ParamData):
                p_write = p.parent_component()
                if p_write not in ComponentSet(parameterList):
                    parameterList.append(p_write)
            else:
                if p not in ComponentSet(parameterList):
                    parameterList.append(p)

        # TODO:  cannot extract this information, waiting on resolution of an issue
        # For now, will raise an error
        raise RuntimeError(
            'Printing of non-models is not currently supported, but will be added soon'
        )
        # setList = identify_components(pyomo_component.expr, pyo.Set)

    else:
        variableList = [
            vr
            for vr in pyomo_component.component_objects(
                pyo.Var, descend_into=True, active=True
            )
        ]

        parameterList = [
            pm
            for pm in pyomo_component.component_objects(
                pyo.Param, descend_into=True, active=True
            )
        ]

        setList = [
            st
            for st in pyomo_component.component_objects(
                pyo.Set, descend_into=True, active=True
            )
        ]

    variableMap = ComponentMap()
    vrIdx = 0
    for i in range(0, len(variableList)):
        vr = variableList[i]
        vrIdx += 1
        if isinstance(vr, ScalarVar):
            variableMap[vr] = 'x_' + str(vrIdx)
        elif isinstance(vr, IndexedVar):
            variableMap[vr] = 'x_' + str(vrIdx)
            for sd in vr.index_set().data():
                vrIdx += 1
                variableMap[vr[sd]] = 'x_' + str(vrIdx)
        else:
            raise DeveloperError(
                'Variable is not a variable.  Should not happen.  Contact developers'
            )

    parameterMap = ComponentMap()
    pmIdx = 0
    for i in range(0, len(parameterList)):
        vr = parameterList[i]
        pmIdx += 1
        if isinstance(vr, ScalarParam):
            parameterMap[vr] = 'p_' + str(pmIdx)
        elif isinstance(vr, IndexedParam):
            parameterMap[vr] = 'p_' + str(pmIdx)
            for sd in vr.index_set().data():
                pmIdx += 1
                parameterMap[vr[sd]] = 'p_' + str(pmIdx)
        else:
            raise DeveloperError(
                'Parameter is not a parameter.  Should not happen.  Contact developers'
            )

    setMap = ComponentMap()
    for i in range(0, len(setList)):
        st = setList[i]
        setMap[st] = 'SET' + str(i + 1)

    # # Only x modes
    # # False : dont use
    # # True  : indexed variables become x_{ix_{subix}}

    if x_only_mode:
        # Need to preserve only the set elements in the overwrite_dict
        new_overwrite_dict = {}
        for ky, vl in overwrite_dict.items():
            if isinstance(ky, _GeneralVarData):
                pass
            elif isinstance(ky, _ParamData):
                pass
            elif isinstance(ky, _SetData):
                new_overwrite_dict[ky] = overwrite_dict[ky]
            else:
                raise ValueError(
                    'The overwrite_dict object has a key of invalid type: %s'
                    % (str(ky))
                )
        overwrite_dict = new_overwrite_dict

        vrIdx = 0
        new_variableMap = ComponentMap()
        for i in range(0, len(variableList)):
            vr = variableList[i]
            vrIdx += 1
            if isinstance(vr, ScalarVar):
                new_variableMap[vr] = 'x_{' + str(vrIdx) + '}'
            elif isinstance(vr, IndexedVar):
                new_variableMap[vr] = 'x_{' + str(vrIdx) + '}'
                for sd in vr.index_set().data():
                    # vrIdx += 1
                    sdString = str(sd)
                    if sdString[0] == '(':
                        sdString = sdString[1:]
                    if sdString[-1] == ')':
                        sdString = sdString[0:-1]
                    new_variableMap[vr[sd]] = (
                        'x_{' + str(vrIdx) + '_{' + sdString + '}' + '}'
                    )
            else:
                raise DeveloperError(
                    'Variable is not a variable.  Should not happen.  Contact developers'
                )

        pmIdx = 0
        new_parameterMap = ComponentMap()
        for i in range(0, len(parameterList)):
            pm = parameterList[i]
            pmIdx += 1
            if isinstance(pm, ScalarParam):
                new_parameterMap[pm] = 'p_{' + str(pmIdx) + '}'
            elif isinstance(pm, IndexedParam):
                new_parameterMap[pm] = 'p_{' + str(pmIdx) + '}'
                for sd in pm.index_set().data():
                    sdString = str(sd)
                    if sdString[0] == '(':
                        sdString = sdString[1:]
                    if sdString[-1] == ')':
                        sdString = sdString[0:-1]
                    new_parameterMap[pm[sd]] = (
                        'p_{' + str(pmIdx) + '_{' + sdString + '}' + '}'
                    )
            else:
                raise DeveloperError(
                    'Parameter is not a parameter.  Should not happen.  Contact developers'
                )

        new_overwrite_dict = ComponentMap()
        for ky, vl in new_variableMap.items():
            new_overwrite_dict[ky] = vl
        for ky, vl in new_parameterMap.items():
            new_overwrite_dict[ky] = vl
        for ky, vl in overwrite_dict.items():
            new_overwrite_dict[ky] = vl
        overwrite_dict = new_overwrite_dict

    else:
        vrIdx = 0
        new_variableMap = ComponentMap()
        for i in range(0, len(variableList)):
            vr = variableList[i]
            vrIdx += 1
            if isinstance(vr, ScalarVar):
                new_variableMap[vr] = vr.name
            elif isinstance(vr, IndexedVar):
                new_variableMap[vr] = vr.name
                for sd in vr.index_set().data():
                    # vrIdx += 1
                    sdString = str(sd)
                    if sdString[0] == '(':
                        sdString = sdString[1:]
                    if sdString[-1] == ')':
                        sdString = sdString[0:-1]
                    if use_smart_variables:
                        new_variableMap[vr[sd]] = applySmartVariables(
                            vr.name + '_' + sdString
                        )
                    else:
                        new_variableMap[vr[sd]] = vr[sd].name
            else:
                raise DeveloperError(
                    'Variable is not a variable.  Should not happen.  Contact developers'
                )

        pmIdx = 0
        new_parameterMap = ComponentMap()
        for i in range(0, len(parameterList)):
            pm = parameterList[i]
            pmIdx += 1
            if isinstance(pm, ScalarParam):
                new_parameterMap[pm] = pm.name
            elif isinstance(pm, IndexedParam):
                new_parameterMap[pm] = pm.name
                for sd in pm.index_set().data():
                    # pmIdx += 1
                    sdString = str(sd)
                    if sdString[0] == '(':
                        sdString = sdString[1:]
                    if sdString[-1] == ')':
                        sdString = sdString[0:-1]
                    if use_smart_variables:
                        new_parameterMap[pm[sd]] = applySmartVariables(
                            pm.name + '_' + sdString
                        )
                    else:
                        new_parameterMap[pm[sd]] = str(pm[sd])  # .name
            else:
                raise DeveloperError(
                    'Parameter is not a parameter.  Should not happen.  Contact developers'
                )

        for ky, vl in new_variableMap.items():
            if ky not in overwrite_dict.keys():
                overwrite_dict[ky] = vl
        for ky, vl in new_parameterMap.items():
            if ky not in overwrite_dict.keys():
                overwrite_dict[ky] = vl

    for ky in overwrite_dict.keys():
        if isinstance(ky, (pyo.Var, pyo.Param)):
            if use_smart_variables and x_only_mode in [0, 3]:
                overwrite_dict[ky] = applySmartVariables(overwrite_dict[ky])
        elif isinstance(ky, (_GeneralVarData, _ParamData)):
            if use_smart_variables and x_only_mode in [3]:
                overwrite_dict[ky] = applySmartVariables(overwrite_dict[ky])
        elif isinstance(ky, _SetData):
            # already handled
            pass
        elif isinstance(ky, (float, int)):
            # happens when immutable parameters are used, do nothing
            pass
        else:
            raise ValueError(
                'The overwrite_dict object has a key of invalid type: %s' % (str(ky))
            )

    for ky, vl in overwrite_dict.items():
        if use_smart_variables:
            pattern = r'_{([^{]*)}_{([^{]*)}'
            replacement = r'_{\1_{\2}}'
            overwrite_dict[ky] = re.sub(pattern, replacement, overwrite_dict[ky])

            pattern = r'_(.)_{([^}]*)}'
            replacement = r'_{\1_{\2}}'
            overwrite_dict[ky] = re.sub(pattern, replacement, overwrite_dict[ky])
        else:
            overwrite_dict[ky] = vl.replace('_', '\\_')

    defaultSetLatexNames = ComponentMap()
    for i in range(0, len(setList)):
        st = setList[i]
        if use_smart_variables:
            chkName = setList[i].name
            if len(chkName) == 1 and chkName.upper() == chkName:
                chkName += '_mathcal'
            defaultSetLatexNames[st] = applySmartVariables(chkName)
        else:
            defaultSetLatexNames[st] = setList[i].name.replace('_', '\\_')

        ## Could be used in the future if someone has a lot of sets
        # defaultSetLatexNames[st] = 'mathcal{' + alphabetStringGenerator(i).upper() + '}'

        if st in overwrite_dict.keys():
            if use_smart_variables:
                defaultSetLatexNames[st] = applySmartVariables(overwrite_dict[st][0])
            else:
                defaultSetLatexNames[st] = overwrite_dict[st][0].replace('_', '\\_')

        defaultSetLatexNames[st] = defaultSetLatexNames[st].replace(
            '\\mathcal', r'\\mathcal'
        )

    for ky, vl in defaultSetLatexNames.items():
        overwrite_dict[ky] = [vl, []]

    return overwrite_dict
