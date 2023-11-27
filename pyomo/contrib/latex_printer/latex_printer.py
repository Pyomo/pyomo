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
import io
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
from pyomo.core.expr.template_expr import (
    NPV_Numeric_GetItemExpression,
    NPV_Structural_GetItemExpression,
    Numeric_GetAttrExpression,
)
from pyomo.core.expr.numeric_expr import NPV_SumExpression, NPV_DivisionExpression
from pyomo.core.base.block import IndexedBlock

from pyomo.core.base.external import _PythonCallbackFunctionID
from pyomo.core.base.enums import SortComponents

from pyomo.core.base.block import _BlockData

from pyomo.repn.util import ExprType

from pyomo.common import DeveloperError

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

from pyomo.common.errors import InfeasibleConstraintException

from pyomo.common.dependencies import numpy as np, numpy_available


def decoder(num, base):
    if int(num) != abs(num):
        # Requiring an integer is nice, but not strictly necessary;
        # the algorithm works for floating point
        raise ValueError("num should be a nonnegative integer")
    if int(base) != abs(base) or not base:
        raise ValueError("base should be a positive integer")
    ans = []
    while 1:
        ans.append(num % base)
        num //= base
        if not num:
            return list(reversed(ans))


def indexCorrector(ixs, base):
    for i in range(0, len(ixs)):
        ix = ixs[i]
        if i + 1 < len(ixs):
            if ixs[i + 1] == 0:
                ixs[i] -= 1
                ixs[i + 1] = base
                if ixs[i] == 0:
                    ixs = indexCorrector(ixs, base)
    return ixs


def alphabetStringGenerator(num):
    alphabet = ['.', 'i', 'j', 'k', 'm', 'n', 'p', 'q', 'r']

    ixs = decoder(num + 1, len(alphabet) - 1)
    pstr = ''
    ixs = indexCorrector(ixs, len(alphabet) - 1)
    for i in range(0, len(ixs)):
        ix = ixs[i]
        pstr += alphabet[ix]
        pstr = pstr.replace('.', '')
    return pstr


def templatize_expression(expr):
    expr, indices = templatize_rule(expr.parent_block(), expr._rule, expr.index_set())
    return (expr, indices)


def templatize_passthrough(con):
    return (con, [])


def precedenceChecker(node, arg1, arg2=None):
    childPrecedence = []
    for a in node.args:
        if hasattr(a, 'PRECEDENCE'):
            if a.PRECEDENCE is None:
                childPrecedence.append(-1)
            else:
                childPrecedence.append(a.PRECEDENCE)
        else:
            childPrecedence.append(-1)

    if hasattr(node, 'PRECEDENCE'):
        precedence = node.PRECEDENCE
    else:
        # Should never hit this
        raise DeveloperError(
            'This error should never be thrown, node does not have a precedence.  Report to developers'
        )

    if childPrecedence[0] > precedence:
        arg1 = ' \\left( ' + arg1 + ' \\right) '

    if arg2 is not None:
        if childPrecedence[1] > precedence:
            arg2 = ' \\left( ' + arg2 + ' \\right) '

    return arg1, arg2


def handle_negation_node(visitor, node, arg1):
    arg1, tsh = precedenceChecker(node, arg1)
    return '-' + arg1


def handle_product_node(visitor, node, arg1, arg2):
    arg1, arg2 = precedenceChecker(node, arg1, arg2)
    return ' '.join([arg1, arg2])


def handle_pow_node(visitor, node, arg1, arg2):
    arg1, arg2 = precedenceChecker(node, arg1, arg2)
    return "%s^{%s}" % (arg1, arg2)


def handle_division_node(visitor, node, arg1, arg2):
    return '\\frac{%s}{%s}' % (arg1, arg2)


def handle_abs_node(visitor, node, arg1):
    return ' \\left| ' + arg1 + ' \\right| '


def handle_unary_node(visitor, node, arg1):
    fcn_handle = node.getname()
    if fcn_handle == 'log10':
        fcn_handle = 'log_{10}'

    if fcn_handle == 'sqrt':
        return '\\sqrt { ' + arg1 + ' }'
    else:
        return '\\' + fcn_handle + ' \\left( ' + arg1 + ' \\right) '


def handle_equality_node(visitor, node, arg1, arg2):
    return arg1 + ' = ' + arg2


def handle_inequality_node(visitor, node, arg1, arg2):
    return arg1 + ' \\leq ' + arg2


def handle_var_node(visitor, node):
    return visitor.variableMap[node]


def handle_num_node(visitor, node):
    if isinstance(node, float):
        if node.is_integer():
            node = int(node)
    return str(node)


def handle_sumExpression_node(visitor, node, *args):
    rstr = args[0]
    for i in range(1, len(args)):
        if args[i][0] == '-':
            rstr += ' - ' + args[i][1:]
        else:
            rstr += ' + ' + args[i]
    return rstr


def handle_monomialTermExpression_node(visitor, node, arg1, arg2):
    if arg1 == '1':
        return arg2
    elif arg1 == '-1':
        return '-' + arg2
    else:
        return arg1 + ' ' + arg2


def handle_named_expression_node(visitor, node, arg1):
    # needed to preserve consistency with the exitNode function call
    # prevents the need to type check in the exitNode function
    return arg1


def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    return arg1 + ' \\leq ' + arg2 + ' \\leq ' + arg3


def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    return 'f_{\\text{exprIf}}(' + arg1 + ',' + arg2 + ',' + arg3 + ')'

    ## Could be handled in the future using cases or similar

    ## Raises not implemented error
    # raise NotImplementedError('Expr_if objects not supported by the Latex Printer')

    ## Puts cases in a bracketed matrix
    # pstr = ''
    # pstr += '\\begin{Bmatrix} '
    # pstr += arg2 + ' , & ' + arg1 + '\\\\ '
    # pstr += arg3 + ' , & \\text{otherwise}' + '\\\\ '
    # pstr += '\\end{Bmatrix}'
    # return pstr


def handle_external_function_node(visitor, node, *args):
    pstr = ''
    visitor.externalFunctionCounter += 1
    pstr += 'f\\_' + str(visitor.externalFunctionCounter) + '('
    for i in range(0, len(args) - 1):
        pstr += args[i]
        if i <= len(args) - 3:
            pstr += ','
        else:
            pstr += ')'
    return pstr


def handle_functionID_node(visitor, node, *args):
    # seems to just be a placeholder empty wrapper object
    return ''


def handle_indexTemplate_node(visitor, node, *args):
    if node._set in ComponentSet(visitor.setMap.keys()):
        # already detected set, do nothing
        pass
    else:
        visitor.setMap[node._set] = 'SET%d' % (len(visitor.setMap.keys()) + 1)

    return '__I_PLACEHOLDER_8675309_GROUP_%s_%s__' % (
        node._group,
        visitor.setMap[node._set],
    )


def handle_numericGetItemExpression_node(visitor, node, *args):
    joinedName = args[0]

    pstr = ''
    pstr += joinedName + '_{'
    for i in range(1, len(args)):
        pstr += args[i]
        if i <= len(args) - 2:
            pstr += ','
        else:
            pstr += '}'
    return pstr


def handle_templateSumExpression_node(visitor, node, *args):
    pstr = ''
    for i in range(0, len(node._iters)):
        pstr += '\\sum_{__S_PLACEHOLDER_8675309_GROUP_%s_%s__} ' % (
            node._iters[i][0]._group,
            visitor.setMap[node._iters[i][0]._set],
        )

    pstr += args[0]

    return pstr


def handle_param_node(visitor, node):
    return visitor.parameterMap[node]


def handle_str_node(visitor, node):
    return "\\mathtt{'" + node.replace('_', '\\_') + "'}"


def handle_npv_structuralGetItemExpression_node(visitor, node, *args):
    joinedName = args[0]

    pstr = ''
    pstr += joinedName + '['
    for i in range(1, len(args)):
        pstr += args[i]
        if i <= len(args) - 2:
            pstr += ','
        else:
            pstr += ']'
    return pstr


def handle_indexedBlock_node(visitor, node, *args):
    return str(node)


def handle_numericGetAttrExpression_node(visitor, node, *args):
    return args[0] + '.' + args[1]


class _LatexVisitor(StreamBasedExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.externalFunctionCounter = 0

        self._operator_handles = {
            ScalarVar: handle_var_node,
            int: handle_num_node,
            float: handle_num_node,
            NegationExpression: handle_negation_node,
            ProductExpression: handle_product_node,
            DivisionExpression: handle_division_node,
            PowExpression: handle_pow_node,
            AbsExpression: handle_abs_node,
            UnaryFunctionExpression: handle_unary_node,
            Expr_ifExpression: handle_exprif_node,
            EqualityExpression: handle_equality_node,
            InequalityExpression: handle_inequality_node,
            RangedExpression: handle_ranged_inequality_node,
            _GeneralExpressionData: handle_named_expression_node,
            ScalarExpression: handle_named_expression_node,
            kernel.expression.expression: handle_named_expression_node,
            kernel.expression.noclone: handle_named_expression_node,
            _GeneralObjectiveData: handle_named_expression_node,
            _GeneralVarData: handle_var_node,
            ScalarObjective: handle_named_expression_node,
            kernel.objective.objective: handle_named_expression_node,
            ExternalFunctionExpression: handle_external_function_node,
            _PythonCallbackFunctionID: handle_functionID_node,
            LinearExpression: handle_sumExpression_node,
            SumExpression: handle_sumExpression_node,
            MonomialTermExpression: handle_monomialTermExpression_node,
            IndexedVar: handle_var_node,
            IndexTemplate: handle_indexTemplate_node,
            Numeric_GetItemExpression: handle_numericGetItemExpression_node,
            TemplateSumExpression: handle_templateSumExpression_node,
            ScalarParam: handle_param_node,
            _ParamData: handle_param_node,
            IndexedParam: handle_param_node,
            NPV_Numeric_GetItemExpression: handle_numericGetItemExpression_node,
            IndexedBlock: handle_indexedBlock_node,
            NPV_Structural_GetItemExpression: handle_npv_structuralGetItemExpression_node,
            str: handle_str_node,
            Numeric_GetAttrExpression: handle_numericGetAttrExpression_node,
            NPV_SumExpression: handle_sumExpression_node,
            NPV_DivisionExpression: handle_division_node,
        }
        if numpy_available:
            self._operator_handles[np.float64] = handle_num_node

    def exitNode(self, node, data):
        try:
            return self._operator_handles[node.__class__](self, node, *data)
        except:
            raise DeveloperError(
                'Latex printer encountered an error when processing type %s, contact the developers'
                % (node.__class__)
            )


def analyze_variable(vr):
    domainMap = {
        'Reals': '\\mathds{R}',
        'PositiveReals': '\\mathds{R}_{> 0}',
        'NonPositiveReals': '\\mathds{R}_{\\leq 0}',
        'NegativeReals': '\\mathds{R}_{< 0}',
        'NonNegativeReals': '\\mathds{R}_{\\geq 0}',
        'Integers': '\\mathds{Z}',
        'PositiveIntegers': '\\mathds{Z}_{> 0}',
        'NonPositiveIntegers': '\\mathds{Z}_{\\leq 0}',
        'NegativeIntegers': '\\mathds{Z}_{< 0}',
        'NonNegativeIntegers': '\\mathds{Z}_{\\geq 0}',
        'Boolean': '\\left\\{ \\text{True} , \\text{False} \\right \\}',
        'Binary': '\\left\\{ 0 , 1 \\right \\}',
        # 'Any': None,
        # 'AnyWithNone': None,
        'EmptySet': '\\varnothing',
        'UnitInterval': '\\mathds{R}',
        'PercentFraction': '\\mathds{R}',
        # 'RealInterval' :        None    ,
        # 'IntegerInterval' :     None    ,
    }

    domainName = vr.domain.name
    varBounds = vr.bounds
    lowerBoundValue = varBounds[0]
    upperBoundValue = varBounds[1]

    if domainName in ['Reals', 'Integers']:
        if lowerBoundValue is not None:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''

        if upperBoundValue is not None:
            upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''

    elif domainName in ['PositiveReals', 'PositiveIntegers']:
        if lowerBoundValue > 0:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ' 0 < '

        if upperBoundValue is not None:
            if upperBoundValue <= 0:
                raise InfeasibleConstraintException(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''

    elif domainName in ['NonPositiveReals', 'NonPositiveIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue > 0:
                raise InfeasibleConstraintException(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            elif lowerBoundValue == 0:
                lowerBound = ' 0 = '
            else:
                lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''

        if upperBoundValue >= 0:
            upperBound = ' \\leq 0 '
        else:
            upperBound = ' \\leq ' + str(upperBoundValue)

    elif domainName in ['NegativeReals', 'NegativeIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue >= 0:
                raise InfeasibleConstraintException(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            else:
                lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''

        if upperBoundValue >= 0:
            upperBound = ' < 0 '
        else:
            upperBound = ' \\leq ' + str(upperBoundValue)

    elif domainName in ['NonNegativeReals', 'NonNegativeIntegers']:
        if lowerBoundValue > 0:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ' 0 \\leq '

        if upperBoundValue is not None:
            if upperBoundValue < 0:
                raise InfeasibleConstraintException(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            elif upperBoundValue == 0:
                upperBound = ' = 0 '
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''

    elif domainName in ['Boolean', 'Binary', 'Any', 'AnyWithNone', 'EmptySet']:
        lowerBound = ''
        upperBound = ''

    elif domainName in ['UnitInterval', 'PercentFraction']:
        if lowerBoundValue > 1:
            raise InfeasibleConstraintException(
                'Formulation is infeasible due to bounds on variable %s' % (vr.name)
            )
        elif lowerBoundValue == 1:
            lowerBound = ' = 1 '
        elif lowerBoundValue > 0:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ' 0 \\leq '

        if upperBoundValue < 0:
            raise InfeasibleConstraintException(
                'Formulation is infeasible due to bounds on variable %s' % (vr.name)
            )
        elif upperBoundValue == 0:
            upperBound = ' = 0 '
        elif upperBoundValue < 1:
            upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ' \\leq 1 '

    else:
        raise NotImplementedError(
            'Invalid domain encountered, will be supported in a future update'
        )

    varBoundData = {
        'variable': vr,
        'lowerBound': lowerBound,
        'upperBound': upperBound,
        'domainName': domainName,
        'domainLatex': domainMap[domainName],
    }

    return varBoundData


def multiple_replace(pstr, rep_dict):
    pattern = re.compile("|".join(rep_dict.keys()), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], pstr)


def latex_printer(
    pyomo_component,
    latex_component_map=None,
    ostream=None,
    use_equation_environment=False,
    explicit_set_summation=False,
    throw_templatization_error=False,
):
    """This function produces a string that can be rendered as LaTeX

    Prints a Pyomo component (Block, Model, Objective, Constraint, or Expression) to a LaTeX compatible string

    Parameters
    ----------
    pyomo_component: _BlockData or Model or Objective or Constraint or Expression
        The Pyomo component to be printed

    latex_component_map: pyomo.common.collections.component_map.ComponentMap
        A map keyed by Pyomo component, values become the LaTeX representation in
        the printer

    ostream: io.TextIOWrapper or io.StringIO or str
        The object to print the LaTeX string to.  Can be an open file object,
        string I/O object, or a string for a filename to write to

    use_equation_environment: bool
        If False, the equation/aligned construction is used to create a single
         LaTeX equation.  If True, then the align environment is used in LaTeX and
         each constraint and objective will be given an individual equation number

    explicit_set_summation: bool
        If False, all sums will be done over 'index in set' or similar.  If True,
        sums will be done over 'i=1' to 'N' or similar if the set is a continuous
        set

    throw_templatization_error: bool
        Option to throw an error on templatization failure rather than
        printing each constraint individually, useful for very large models


    Returns
    -------
    str
        A LaTeX string of the pyomo_component

    """

    # Various setup things

    # is Single implies Objective, constraint, or expression
    # these objects require a slight modification of behavior
    # isSingle==False means a model or block

    use_short_descriptors = True

    # Cody's backdoor because he got outvoted
    if latex_component_map is not None:
        if 'use_short_descriptors' in list(latex_component_map.keys()):
            if latex_component_map['use_short_descriptors'] == False:
                use_short_descriptors = False

    if latex_component_map is None:
        latex_component_map = ComponentMap()
        existing_components = ComponentSet([])
    else:
        existing_components = ComponentSet(list(latex_component_map.keys()))

    isSingle = False

    if isinstance(pyomo_component, pyo.Objective):
        objectives = [pyomo_component]
        constraints = []
        expressions = []
        templatize_fcn = templatize_constraint
        use_equation_environment = True
        isSingle = True

    elif isinstance(pyomo_component, pyo.Constraint):
        objectives = []
        constraints = [pyomo_component]
        expressions = []
        templatize_fcn = templatize_constraint
        use_equation_environment = True
        isSingle = True

    elif isinstance(pyomo_component, pyo.Expression):
        objectives = []
        constraints = []
        expressions = [pyomo_component]
        templatize_fcn = templatize_expression
        use_equation_environment = True
        isSingle = True

    elif isinstance(pyomo_component, (ExpressionBase, pyo.Var)):
        objectives = []
        constraints = []
        expressions = [pyomo_component]
        templatize_fcn = templatize_passthrough
        use_equation_environment = True
        isSingle = True

    elif isinstance(pyomo_component, _BlockData):
        objectives = [
            obj
            for obj in pyomo_component.component_data_objects(
                pyo.Objective,
                descend_into=True,
                active=True,
                sort=SortComponents.deterministic,
            )
        ]
        constraints = [
            con
            for con in pyomo_component.component_objects(
                pyo.Constraint,
                descend_into=True,
                active=True,
                sort=SortComponents.deterministic,
            )
        ]
        expressions = []
        templatize_fcn = templatize_constraint

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

        # Will grab the sets as the expression is walked
        setList = []

    else:
        variableList = [
            vr
            for vr in pyomo_component.component_objects(
                pyo.Var,
                descend_into=True,
                active=True,
                sort=SortComponents.deterministic,
            )
        ]

        parameterList = [
            pm
            for pm in pyomo_component.component_objects(
                pyo.Param,
                descend_into=True,
                active=True,
                sort=SortComponents.deterministic,
            )
        ]

        setList = [
            st
            for st in pyomo_component.component_objects(
                pyo.Set,
                descend_into=True,
                active=True,
                sort=SortComponents.deterministic,
            )
        ]

    descriptorDict = {}
    if use_short_descriptors:
        descriptorDict['minimize'] = '\\min'
        descriptorDict['maximize'] = '\\max'
        descriptorDict['subject to'] = '\\text{s.t.}'
        descriptorDict['with bounds'] = '\\text{w.b.}'
    else:
        descriptorDict['minimize'] = '\\text{minimize}'
        descriptorDict['maximize'] = '\\text{maximize}'
        descriptorDict['subject to'] = '\\text{subject to}'
        descriptorDict['with bounds'] = '\\text{with bounds}'

    # In the case where just a single expression is passed, add this to the constraint list for printing
    constraints = constraints + expressions

    # Declare a visitor/walker
    visitor = _LatexVisitor()

    variableMap = ComponentMap()
    vrIdx = 0
    for vr in variableList:
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
    visitor.variableMap = variableMap

    parameterMap = ComponentMap()
    pmIdx = 0
    for vr in parameterList:
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
    visitor.parameterMap = parameterMap

    setMap = ComponentMap()
    for i in range(0, len(setList)):
        st = setList[i]
        setMap[st] = 'SET' + str(i + 1)
    visitor.setMap = setMap

    # starts building the output string
    pstr = ''
    if not use_equation_environment:
        pstr += '\\begin{align} \n'
        tbSpc = 4
        trailingAligner = '& '
    else:
        pstr += '\\begin{equation} \n'
        if not isSingle:
            pstr += '    \\begin{aligned} \n'
            tbSpc = 8
        else:
            tbSpc = 4
        trailingAligner = ''

    # Iterate over the objectives and print
    for obj in objectives:
        try:
            obj_template, obj_indices = templatize_fcn(obj)
        except:
            if throw_templatization_error:
                raise RuntimeError(
                    "An objective named '%s' has been constructed that cannot be templatized"
                    % (obj.__str__())
                )
            else:
                obj_template = obj

        if obj.sense == pyo.minimize:  # or == 1
            pstr += ' ' * tbSpc + '& %s \n' % (descriptorDict['minimize'])
        else:
            pstr += ' ' * tbSpc + '& %s \n' % (descriptorDict['maximize'])

        pstr += ' ' * tbSpc + '& & %s %s' % (
            visitor.walk_expression(obj_template),
            trailingAligner,
        )
        if not use_equation_environment:
            pstr += '\\label{obj:' + pyomo_component.name + '_' + obj.name + '} '
        if not isSingle:
            pstr += '\\\\ \n'
        else:
            pstr += '\n'

    # Iterate over the constraints
    if len(constraints) > 0:
        # only print this if printing a full formulation
        if not isSingle:
            pstr += ' ' * tbSpc + '& %s \n' % (descriptorDict['subject to'])

        # first constraint needs different alignment because of the 'subject to':
        # & minimize   & & [Objective]
        # & subject to & & [Constraint 1]
        # &            & & [Constraint 2]
        # &            & & [Constraint N]

        # The double '& &' renders better for some reason

        for i, con in enumerate(constraints):
            if not isSingle:
                if i == 0:
                    algn = '& &'
                else:
                    algn = '&&&'
            else:
                algn = ''

            if not isSingle:
                tail = '\\\\ \n'
            else:
                tail = '\n'

            # grab the constraint and templatize
            try:
                con_template, indices = templatize_fcn(con)
                con_template_list = [con_template]
            except:
                if throw_templatization_error:
                    raise RuntimeError(
                        "A constraint named '%s' has been constructed that cannot be templatized"
                        % (con.__str__())
                    )
                else:
                    con_template_list = [c.expr for c in con.values()]
                    indices = []

            for con_template in con_template_list:
                # Walk the constraint
                conLine = (
                    ' ' * tbSpc
                    + algn
                    + ' %s %s'
                    % (visitor.walk_expression(con_template), trailingAligner)
                )

                # setMap = visitor.setMap
                # Multiple constraints are generated using a set
                if len(indices) > 0:
                    if indices[0]._set in ComponentSet(visitor.setMap.keys()):
                        # already detected set, do nothing
                        pass
                    else:
                        visitor.setMap[indices[0]._set] = 'SET%d' % (
                            len(visitor.setMap.keys()) + 1
                        )

                    idxTag = '__I_PLACEHOLDER_8675309_GROUP_%s_%s__' % (
                        indices[0]._group,
                        visitor.setMap[indices[0]._set],
                    )
                    setTag = '__S_PLACEHOLDER_8675309_GROUP_%s_%s__' % (
                        indices[0]._group,
                        visitor.setMap[indices[0]._set],
                    )

                    conLine += ' \\qquad \\forall %s \\in %s ' % (idxTag, setTag)
                pstr += conLine

                # Add labels as needed
                if not use_equation_environment:
                    pstr += (
                        '\\label{con:' + pyomo_component.name + '_' + con.name + '} '
                    )

                pstr += tail

    # Print bounds and sets
    if not isSingle:
        varBoundData = []
        for i in range(0, len(variableList)):
            vr = variableList[i]
            if isinstance(vr, ScalarVar):
                varBoundDataEntry = analyze_variable(vr)
                varBoundData.append(varBoundDataEntry)
            elif isinstance(vr, IndexedVar):
                varBoundData_indexedVar = []
                setData = vr.index_set().data()
                for sd in setData:
                    varBoundDataEntry = analyze_variable(vr[sd])
                    varBoundData_indexedVar.append(varBoundDataEntry)
                globIndexedVariables = True
                for j in range(0, len(varBoundData_indexedVar) - 1):
                    chks = []
                    chks.append(
                        varBoundData_indexedVar[j]['lowerBound']
                        == varBoundData_indexedVar[j + 1]['lowerBound']
                    )
                    chks.append(
                        varBoundData_indexedVar[j]['upperBound']
                        == varBoundData_indexedVar[j + 1]['upperBound']
                    )
                    chks.append(
                        varBoundData_indexedVar[j]['domainName']
                        == varBoundData_indexedVar[j + 1]['domainName']
                    )
                    if not all(chks):
                        globIndexedVariables = False
                        break
                if globIndexedVariables:
                    varBoundData.append(
                        {
                            'variable': vr,
                            'lowerBound': varBoundData_indexedVar[0]['lowerBound'],
                            'upperBound': varBoundData_indexedVar[0]['upperBound'],
                            'domainName': varBoundData_indexedVar[0]['domainName'],
                            'domainLatex': varBoundData_indexedVar[0]['domainLatex'],
                        }
                    )
                else:
                    varBoundData += varBoundData_indexedVar
            else:
                raise DeveloperError(
                    'Variable is not a variable.  Should not happen.  Contact developers'
                )

        # print the accumulated data to the string
        bstr = ''
        appendBoundString = False
        useThreeAlgn = False
        for i, vbd in enumerate(varBoundData):
            if (
                vbd['lowerBound'] == ''
                and vbd['upperBound'] == ''
                and vbd['domainName'] == 'Reals'
            ):
                # unbounded all real, do not print
                if i == len(varBoundData) - 1:
                    bstr = bstr[0:-4]
            else:
                if not useThreeAlgn:
                    algn = '& &'
                    useThreeAlgn = True
                else:
                    algn = '&&&'

                if use_equation_environment:
                    conLabel = ''
                else:
                    conLabel = (
                        ' \\label{con:'
                        + pyomo_component.name
                        + '_'
                        + variableMap[vbd['variable']]
                        + '_bound'
                        + '} '
                    )

                appendBoundString = True
                coreString = (
                    vbd['lowerBound']
                    + variableMap[vbd['variable']]
                    + vbd['upperBound']
                    + ' '
                    + trailingAligner
                    + '\\qquad \\in '
                    + vbd['domainLatex']
                    + conLabel
                )
                bstr += ' ' * tbSpc + algn + ' %s' % (coreString)
                if i <= len(varBoundData) - 2:
                    bstr += '\\\\ \n'
                else:
                    bstr += '\n'

        if appendBoundString:
            pstr += ' ' * tbSpc + '& %s \n' % (descriptorDict['with bounds'])
            pstr += bstr + '\n'
        else:
            pstr = pstr[0:-4] + '\n'

    # close off the print string
    if not use_equation_environment:
        pstr += '\\end{align} \n'
    else:
        if not isSingle:
            pstr += '    \\end{aligned} \n'
            pstr += '    \\label{%s} \n' % (pyomo_component.name)
        pstr += '\\end{equation} \n'

    setMap = visitor.setMap
    setMap_inverse = {vl: ky for ky, vl in setMap.items()}

    # Handling the iterator indices
    defaultSetLatexNames = ComponentMap()
    for ky, vl in setMap.items():
        st = ky
        defaultSetLatexNames[st] = st.name.replace('_', '\\_')
        if st in ComponentSet(latex_component_map.keys()):
            defaultSetLatexNames[st] = latex_component_map[st][
                0
            ]  # .replace('_', '\\_')

    latexLines = pstr.split('\n')
    for jj in range(0, len(latexLines)):
        groupMap = {}
        uniqueSets = []
        ln = latexLines[jj]
        # only modify if there is a placeholder in the line
        if "PLACEHOLDER_8675309_GROUP_" in ln:
            splitLatex = ln.split('__')
            # Find the unique combinations of group numbers and set names
            for word in splitLatex:
                if "PLACEHOLDER_8675309_GROUP_" in word:
                    ifo = word.split("PLACEHOLDER_8675309_GROUP_")[1]
                    gpNum, stName = ifo.split('_')
                    if gpNum not in groupMap.keys():
                        groupMap[gpNum] = [stName]
                    if stName not in ComponentSet(uniqueSets):
                        uniqueSets.append(stName)

            # Determine if the set is continuous
            setInfo = dict(
                zip(
                    uniqueSets,
                    [{'continuous': False} for i in range(0, len(uniqueSets))],
                )
            )

            for ky, vl in setInfo.items():
                ix = int(ky[3:]) - 1
                setInfo[ky]['setObject'] = setMap_inverse[ky]  # setList[ix]
                setInfo[ky][
                    'setRegEx'
                ] = r'__S_PLACEHOLDER_8675309_GROUP_([0-9*])_%s__' % (ky)
                setInfo[ky][
                    'sumSetRegEx'
                ] = r'sum_{__S_PLACEHOLDER_8675309_GROUP_([0-9*])_%s__}' % (ky)
                # setInfo[ky]['idxRegEx'] = r'__I_PLACEHOLDER_8675309_GROUP_[0-9*]_%s__'%(ky)

            if explicit_set_summation:
                for ky, vl in setInfo.items():
                    st = vl['setObject']
                    stData = st.data()
                    stCont = True
                    for ii in range(0, len(stData)):
                        if ii + stData[0] != stData[ii]:
                            stCont = False
                            break
                    setInfo[ky]['continuous'] = stCont

            # replace the sets
            for ky, vl in setInfo.items():
                # if the set is continuous and the flag has been set
                if explicit_set_summation and setInfo[ky]['continuous']:
                    st = setInfo[ky]['setObject']
                    stData = st.data()
                    bgn = stData[0]
                    ed = stData[-1]

                    replacement = (
                        r'sum_{ __I_PLACEHOLDER_8675309_GROUP_\1_%s__ = %d }^{%d}'
                        % (ky, bgn, ed)
                    )
                    ln = re.sub(setInfo[ky]['sumSetRegEx'], replacement, ln)
                else:
                    # if the set is not continuous or the flag has not been set
                    replacement = (
                        r'sum_{ __I_PLACEHOLDER_8675309_GROUP_\1_%s__ \\in __S_PLACEHOLDER_8675309_GROUP_\1_%s__  }'
                        % (ky, ky)
                    )
                    ln = re.sub(setInfo[ky]['sumSetRegEx'], replacement, ln)

                replacement = repr(defaultSetLatexNames[setInfo[ky]['setObject']])[1:-1]
                ln = re.sub(setInfo[ky]['setRegEx'], replacement, ln)

            # groupNumbers = re.findall(r'__I_PLACEHOLDER_8675309_GROUP_([0-9*])_SET[0-9]*__',ln)
            setNumbers = re.findall(
                r'__I_PLACEHOLDER_8675309_GROUP_[0-9*]_SET([0-9]*)__', ln
            )
            groupSetPairs = re.findall(
                r'__I_PLACEHOLDER_8675309_GROUP_([0-9*])_SET([0-9]*)__', ln
            )

            groupInfo = {}
            for vl in setNumbers:
                groupInfo['SET' + vl] = {
                    'setObject': setInfo['SET' + vl]['setObject'],
                    'indices': [],
                }

            for gp in groupSetPairs:
                if gp[0] not in groupInfo['SET' + gp[1]]['indices']:
                    groupInfo['SET' + gp[1]]['indices'].append(gp[0])

            indexCounter = 0
            for ky, vl in groupInfo.items():
                if vl['setObject'] in ComponentSet(latex_component_map.keys()):
                    indexNames = latex_component_map[vl['setObject']][1]
                    if len(indexNames) != 0:
                        if len(indexNames) < len(vl['indices']):
                            raise ValueError(
                                'Insufficient number of indices provided to the overwrite dictionary for set %s'
                                % (vl['setObject'].name)
                            )
                        for i in range(0, len(vl['indices'])):
                            ln = ln.replace(
                                '__I_PLACEHOLDER_8675309_GROUP_%s_%s__'
                                % (vl['indices'][i], ky),
                                indexNames[i],
                            )
                    else:
                        for i in range(0, len(vl['indices'])):
                            ln = ln.replace(
                                '__I_PLACEHOLDER_8675309_GROUP_%s_%s__'
                                % (vl['indices'][i], ky),
                                alphabetStringGenerator(indexCounter),
                            )
                            indexCounter += 1
                else:
                    for i in range(0, len(vl['indices'])):
                        ln = ln.replace(
                            '__I_PLACEHOLDER_8675309_GROUP_%s_%s__'
                            % (vl['indices'][i], ky),
                            alphabetStringGenerator(indexCounter),
                        )
                        indexCounter += 1

        latexLines[jj] = ln

    pstr = '\n'.join(latexLines)

    new_variableMap = ComponentMap()
    for i, vr in enumerate(variableList):
        if isinstance(vr, ScalarVar):
            new_variableMap[vr] = vr.name
        elif isinstance(vr, IndexedVar):
            new_variableMap[vr] = vr.name
            for sd in vr.index_set().data():
                sdString = str(sd)
                if sdString[0] == '(':
                    sdString = sdString[1:]
                if sdString[-1] == ')':
                    sdString = sdString[0:-1]
                new_variableMap[vr[sd]] = vr[sd].name
        else:
            raise DeveloperError(
                'Variable is not a variable.  Should not happen.  Contact developers'
            )

    new_parameterMap = ComponentMap()
    for i, pm in enumerate(parameterList):
        pm = parameterList[i]
        if isinstance(pm, ScalarParam):
            new_parameterMap[pm] = pm.name
        elif isinstance(pm, IndexedParam):
            new_parameterMap[pm] = pm.name
            for sd in pm.index_set().data():
                sdString = str(sd)
                if sdString[0] == '(':
                    sdString = sdString[1:]
                if sdString[-1] == ')':
                    sdString = sdString[0:-1]
                new_parameterMap[pm[sd]] = str(pm[sd])  # .name
        else:
            raise DeveloperError(
                'Parameter is not a parameter.  Should not happen.  Contact developers'
            )

    for ky, vl in new_variableMap.items():
        if ky not in ComponentSet(latex_component_map.keys()):
            latex_component_map[ky] = vl
    for ky, vl in new_parameterMap.items():
        if ky not in ComponentSet(latex_component_map.keys()):
            latex_component_map[ky] = vl

    rep_dict = {}
    for ky in ComponentSet(list(reversed(list(latex_component_map.keys())))):
        if isinstance(ky, (pyo.Var, _GeneralVarData)):
            overwrite_value = latex_component_map[ky]
            if ky not in existing_components:
                overwrite_value = overwrite_value.replace('_', '\\_')
            rep_dict[variableMap[ky]] = overwrite_value
        elif isinstance(ky, (pyo.Param, _ParamData)):
            overwrite_value = latex_component_map[ky]
            if ky not in existing_components:
                overwrite_value = overwrite_value.replace('_', '\\_')
            rep_dict[parameterMap[ky]] = overwrite_value
        elif isinstance(ky, _SetData):
            # already handled
            pass
        elif isinstance(ky, (float, int)):
            # happens when immutable parameters are used, do nothing
            pass
        else:
            raise ValueError(
                'The latex_component_map object has a key of invalid type: %s'
                % (str(ky))
            )

    label_rep_dict = copy.deepcopy(rep_dict)
    for ky, vl in label_rep_dict.items():
        label_rep_dict[ky] = vl.replace('{', '').replace('}', '').replace('\\', '')

    splitLines = pstr.split('\n')
    for i in range(0, len(splitLines)):
        if use_equation_environment:
            splitLines[i] = multiple_replace(splitLines[i], rep_dict)
        else:
            if '\\label{' in splitLines[i]:
                epr, lbl = splitLines[i].split('\\label{')
                epr = multiple_replace(epr, rep_dict)
                # rep_dict[ky] = vl.replace('_', '\\_')
                lbl = multiple_replace(lbl, label_rep_dict)
                splitLines[i] = epr + '\\label{' + lbl

    pstr = '\n'.join(splitLines)

    pattern = r'_{([^{]*)}_{([^{]*)}'
    replacement = r'_{\1_{\2}}'
    pstr = re.sub(pattern, replacement, pstr)

    pattern = r'_(.)_{([^}]*)}'
    replacement = r'_{\1_{\2}}'
    pstr = re.sub(pattern, replacement, pstr)

    splitLines = pstr.split('\n')
    finalLines = []
    for sl in splitLines:
        if sl != '':
            finalLines.append(sl)

    pstr = '\n'.join(finalLines)

    if ostream is not None:
        fstr = ''
        fstr += '\\documentclass{article} \n'
        fstr += '\\usepackage{amsmath} \n'
        fstr += '\\usepackage{amssymb} \n'
        fstr += '\\usepackage{dsfont} \n'
        fstr += '\\usepackage[paperheight=11in, paperwidth=8.5in, left=1in, right=1in, top=1in, bottom=1in]{geometry} \n'
        fstr += '\\allowdisplaybreaks \n'
        fstr += '\\begin{document} \n'
        fstr += '\\normalsize \n'
        fstr += pstr + '\n'
        fstr += '\\end{document} \n'

        # optional write to output file
        if isinstance(ostream, (io.TextIOWrapper, io.StringIO)):
            ostream.write(fstr)
        elif isinstance(ostream, str):
            f = open(ostream, 'w')
            f.write(fstr)
            f.close()
        else:
            raise ValueError(
                'Invalid type %s encountered when parsing the ostream.  Must be a StringIO, FileIO, or valid filename string'
            )

    # return the latex string
    return pstr
