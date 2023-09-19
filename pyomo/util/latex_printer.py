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


def decoder(num, base):
    if isinstance(base, float):
        if not base.is_integer():
            raise ValueError('Invalid base')
        else:
            base = int(base)

    if base <= 1:
        raise ValueError('Invalid base')

    if num == 0:
        numDigs = 1
    else:
        numDigs = math.ceil(math.log(num, base))
        if math.log(num, base).is_integer():
            numDigs += 1
    digs = [0.0 for i in range(0, numDigs)]
    rem = num
    for i in range(0, numDigs):
        ix = numDigs - i - 1
        dg = math.floor(rem / base**ix)
        rem = rem % base**ix
        digs[i] = dg
    return digs


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


def alphabetStringGenerator(num, indexMode=False):
    if indexMode:
        alphabet = [
            '.',
            'i',
            'j',
            'k',
            'm',
            'n',
            'p',
            'q',
            'r',
            # 'a',
            # 'b',
            # 'c',
            # 'd',
            # 'e',
            # 'f',
            # 'g',
            # 'h',
            # 'l',
            # 'o',
            # 's',
            # 't',
            # 'u',
            # 'v',
            # 'w',
            # 'x',
            # 'y',
            # 'z',
        ]

    else:
        alphabet = [
            '.',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
        ]
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
    # needed to preserve consistencency with the exitNode function call
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
    pstr += 'f('
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
    return '__I_PLACEHOLDER_8675309_GROUP_%s_%s__' % (
        node._group,
        visitor.setMap[node._set],
    )


def handle_numericGIE_node(visitor, node, *args):
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


class _LatexVisitor(StreamBasedExpressionVisitor):
    def __init__(self):
        super().__init__()

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
            Numeric_GetItemExpression: handle_numericGIE_node,
            TemplateSumExpression: handle_templateSumExpression_node,
            ScalarParam: handle_param_node,
            _ParamData: handle_param_node,
        }

    def exitNode(self, node, data):
        return self._operator_handles[node.__class__](self, node, *data)


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


def analyze_variable(vr, visitor):
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
        'Boolean': '\\left\\{ 0 , 1 \\right \\}',
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
        if lowerBoundValue is not None:
            if lowerBoundValue > 0:
                lowerBound = str(lowerBoundValue) + ' \\leq '
            else:
                lowerBound = ' 0 < '
        else:
            lowerBound = ' 0 < '

        if upperBoundValue is not None:
            if upperBoundValue <= 0:
                raise ValueError(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''

    elif domainName in ['NonPositiveReals', 'NonPositiveIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue > 0:
                raise ValueError(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            elif lowerBoundValue == 0:
                lowerBound = ' 0 = '
            else:
                lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''

        if upperBoundValue is not None:
            if upperBoundValue >= 0:
                upperBound = ' \\leq 0 '
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ' \\leq 0 '

    elif domainName in ['NegativeReals', 'NegativeIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue >= 0:
                raise ValueError(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            else:
                lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''

        if upperBoundValue is not None:
            if upperBoundValue >= 0:
                upperBound = ' < 0 '
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ' < 0 '

    elif domainName in ['NonNegativeReals', 'NonNegativeIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue > 0:
                lowerBound = str(lowerBoundValue) + ' \\leq '
            else:
                lowerBound = ' 0 \\leq '
        else:
            lowerBound = ' 0 \\leq '

        if upperBoundValue is not None:
            if upperBoundValue < 0:
                raise ValueError(
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
        if lowerBoundValue is not None:
            if lowerBoundValue > 0:
                lowerBound = str(lowerBoundValue) + ' \\leq '
            elif lowerBoundValue > 1:
                raise ValueError(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            elif lowerBoundValue == 1:
                lowerBound = ' = 1 '
            else:
                lowerBound = ' 0 \\leq '
        else:
            lowerBound = ' 0 \\leq '

        if upperBoundValue is not None:
            if upperBoundValue < 1:
                upperBound = ' \\leq ' + str(upperBoundValue)
            elif upperBoundValue < 0:
                raise ValueError(
                    'Formulation is infeasible due to bounds on variable %s' % (vr.name)
                )
            elif upperBoundValue == 0:
                upperBound = ' = 0 '
            else:
                upperBound = ' \\leq 1 '
        else:
            upperBound = ' \\leq 1 '

    else:
        raise ValueError('Domain %s not supported by the latex printer' % (domainName))

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
    filename=None,
    use_equation_environment=False,
    split_continuous_sets=False,
    use_smart_variables=False,
    x_only_mode=0,
    use_short_descriptors=False,
    overwrite_dict=None,
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
                pyo.Objective, descend_into=True, active=True
            )
        ]
        constraints = [
            con
            for con in pyomo_component.component_objects(
                pyo.Constraint, descend_into=True, active=True
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

        # TODO:  cannot extract this information, waiting on resolution of an issue
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

    forallTag = ' \\qquad \\forall'

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
    visitor.variableMap = variableMap

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
        trailingAligner = '&'

    # Iterate over the objectives and print
    for obj in objectives:
        try:
            obj_template, obj_indices = templatize_fcn(obj)
        except:
            raise RuntimeError(
                "An objective has been constructed that cannot be templatized"
            )

        if obj.sense == 1:
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

        for i in range(0, len(constraints)):
            if not isSingle:
                if i == 0:
                    algn = '& &'
                else:
                    algn = '&&&'
            else:
                algn = ''

            tail = '\\\\ \n'

            # grab the constraint and templatize
            con = constraints[i]
            try:
                con_template, indices = templatize_fcn(con)
            except:
                raise RuntimeError(
                    "A constraint has been constructed that cannot be templatized"
                )

            # Walk the constraint
            conLine = (
                ' ' * tbSpc
                + algn
                + ' %s %s' % (visitor.walk_expression(con_template), trailingAligner)
            )

            # Multiple constraints are generated using a set
            if len(indices) > 0:
                idxTag = '__I_PLACEHOLDER_8675309_GROUP_%s_%s__' % (
                    indices[0]._group,
                    setMap[indices[0]._set],
                )
                setTag = '__S_PLACEHOLDER_8675309_GROUP_%s_%s__' % (
                    indices[0]._group,
                    setMap[indices[0]._set],
                )

                conLine += '%s %s \\in %s ' % (forallTag, idxTag, setTag)
            pstr += conLine

            # Add labels as needed
            if not use_equation_environment:
                pstr += '\\label{con:' + pyomo_component.name + '_' + con.name + '} '

            # prevents an emptly blank line from being at the end of the latex output
            if i <= len(constraints) - 2:
                pstr += tail
            else:
                pstr += tail
                # pstr += '\n'

    # Print bounds and sets
    if not isSingle:
        variableList = [
            vr
            for vr in pyomo_component.component_objects(
                pyo.Var, descend_into=True, active=True
            )
        ]

        varBoundData = []
        for i in range(0, len(variableList)):
            vr = variableList[i]
            if isinstance(vr, ScalarVar):
                varBoundDataEntry = analyze_variable(vr, visitor)
                varBoundData.append(varBoundDataEntry)
            elif isinstance(vr, IndexedVar):
                varBoundData_indexedVar = []
                # need to wrap in function and do individually
                # Check on the final variable after all the indices are processed
                setData = vr.index_set().data()
                for sd in setData:
                    varBoundDataEntry = analyze_variable(vr[sd], visitor)
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
        for i in range(0, len(varBoundData)):
            vbd = varBoundData[i]
            if (
                vbd['lowerBound'] == ''
                and vbd['upperBound'] == ''
                and vbd['domainName'] == 'Reals'
            ):
                # unbounded all real, do not print
                if i <= len(varBoundData) - 2:
                    bstr = bstr[0:-2]
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

    # Handling the iterator indices
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
                    if stName not in uniqueSets:
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
                setInfo[ky]['setObject'] = setList[ix]
                setInfo[ky][
                    'setRegEx'
                ] = r'__S_PLACEHOLDER_8675309_GROUP_([0-9*])_%s__' % (ky)
                setInfo[ky][
                    'sumSetRegEx'
                ] = r'sum_{__S_PLACEHOLDER_8675309_GROUP_([0-9*])_%s__}' % (ky)
                # setInfo[ky]['idxRegEx'] = r'__I_PLACEHOLDER_8675309_GROUP_[0-9*]_%s__'%(ky)

            if split_continuous_sets:
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
                if split_continuous_sets and setInfo[ky]['continuous']:
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

                replacement = defaultSetLatexNames[setInfo[ky]['setObject']]
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
                if vl['setObject'] in overwrite_dict.keys():
                    indexNames = overwrite_dict[vl['setObject']][1]
                    if len(indexNames) < len(vl['indices']):
                        raise ValueError(
                            'Insufficient number of indices provided to the overwrite dictionary for set %s'
                            % (vl['setObject'].name)
                        )
                    for i in range(0, len(indexNames)):
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
                            alphabetStringGenerator(indexCounter, True),
                        )
                        indexCounter += 1

            # print('gn',groupInfo)

        latexLines[jj] = ln

    pstr = '\n'.join(latexLines)
    # pstr = pstr.replace('\\mathcal{', 'mathcal{')
    # pstr = pstr.replace('mathcal{', '\\mathcal{')

    if x_only_mode in [1, 2, 3]:
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

    # # Only x modes
    # # Mode 0 : dont use
    # # Mode 1 : indexed variables become x_{_{ix}}
    # # Mode 2 : uses standard alphabet [a,...,z,aa,...,az,...,aaa,...] with subscripts for indices, ex: abcd_{ix}
    # # Mode 3 : unwrap everything into an x_{} list, including the indexed vars themselves

    if x_only_mode == 1:
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

    elif x_only_mode == 2:
        vrIdx = 0
        new_variableMap = ComponentMap()
        for i in range(0, len(variableList)):
            vr = variableList[i]
            vrIdx += 1
            if isinstance(vr, ScalarVar):
                new_variableMap[vr] = alphabetStringGenerator(i)
            elif isinstance(vr, IndexedVar):
                new_variableMap[vr] = alphabetStringGenerator(i)
                for sd in vr.index_set().data():
                    # vrIdx += 1
                    sdString = str(sd)
                    if sdString[0] == '(':
                        sdString = sdString[1:]
                    if sdString[-1] == ')':
                        sdString = sdString[0:-1]
                    new_variableMap[vr[sd]] = (
                        alphabetStringGenerator(i) + '_{' + sdString + '}'
                    )
            else:
                raise DeveloperError(
                    'Variable is not a variable.  Should not happen.  Contact developers'
                )

        pmIdx = vrIdx - 1
        new_parameterMap = ComponentMap()
        for i in range(0, len(parameterList)):
            pm = parameterList[i]
            pmIdx += 1
            if isinstance(pm, ScalarParam):
                new_parameterMap[pm] = alphabetStringGenerator(pmIdx)
            elif isinstance(pm, IndexedParam):
                new_parameterMap[pm] = alphabetStringGenerator(pmIdx)
                for sd in pm.index_set().data():
                    sdString = str(sd)
                    if sdString[0] == '(':
                        sdString = sdString[1:]
                    if sdString[-1] == ')':
                        sdString = sdString[0:-1]
                    new_parameterMap[pm[sd]] = (
                        alphabetStringGenerator(pmIdx) + '_{' + sdString + '}'
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

    elif x_only_mode == 3:
        new_overwrite_dict = ComponentMap()
        for ky, vl in variableMap.items():
            new_overwrite_dict[ky] = vl
        for ky, vl in parameterMap.items():
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

    rep_dict = {}
    for ky in list(reversed(list(overwrite_dict.keys()))):
        if isinstance(ky, (pyo.Var, pyo.Param)):
            if use_smart_variables and x_only_mode in [0, 3]:
                overwrite_value = applySmartVariables(overwrite_dict[ky])
            else:
                overwrite_value = overwrite_dict[ky]
            rep_dict[variableMap[ky]] = overwrite_value
        elif isinstance(ky, (_GeneralVarData, _ParamData)):
            if use_smart_variables and x_only_mode in [3]:
                overwrite_value = applySmartVariables(overwrite_dict[ky])
            else:
                overwrite_value = overwrite_dict[ky]
            rep_dict[variableMap[ky]] = overwrite_value
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

    if not use_smart_variables:
        for ky, vl in rep_dict.items():
            rep_dict[ky] = vl.replace('_', '\\_')

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

    # optional write to output file
    if filename is not None:
        fstr = ''
        fstr += '\\documentclass{article} \n'
        fstr += '\\usepackage{amsmath} \n'
        fstr += '\\usepackage{amssymb} \n'
        fstr += '\\usepackage{dsfont} \n'
        fstr += '\\allowdisplaybreaks \n'
        fstr += '\\begin{document} \n'
        fstr += pstr
        fstr += '\\end{document} \n'
        f = open(filename, 'w')
        f.write(fstr)
        f.close()

    # return the latex string
    return pstr
