import pyomo.environ as pe
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

from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
    GetItemExpression,
    GetAttrExpression,
    TemplateSumExpression,
    IndexTemplate,
)
from pyomo.core.expr.template_expr import Numeric_GetItemExpression
from pyomo.core.expr.template_expr import templatize_constraint, resolve_template
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint

from pyomo.core.base.external import _PythonCallbackFunctionID

from pyomo.core.base.block import _BlockData

from pyomo.repn.util import ExprType

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

# see:  https://github.com/Pyomo/pyomo/blob/main/pyomo/repn/plugins/nl_writer.py


def templatize_expression(expr):
    expr, indices = templatize_rule(expr.parent_block(), expr._rule, expr.index_set())
    return (expr, indices)


def templatize_passthrough(con):
    return (con, [])


def handle_negation_node(node, arg1):
    return '-' + arg1


def handle_product_node(node, arg1, arg2):
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
        precedence = -1

    if childPrecedence[0] > precedence:
        arg1 = ' \\left( ' + arg1 + ' \\right) '

    if childPrecedence[1] > precedence:
        arg2 = ' \\left( ' + arg2 + ' \\right) '

    return ''.join([arg1, arg2])


def handle_division_node(node, arg1, arg2):
    # return '/'.join([arg1,arg2])
    return '\\frac{%s}{%s}' % (arg1, arg2)


def handle_pow_node(node, arg1, arg2):
    return "%s^{%s}" % (arg1, arg2)


def handle_abs_node(node, arg1):
    return ' \\left| ' + arg1 + ' \\right| '


def handle_unary_node(node, arg1):
    fcn_handle = node.getname()
    if fcn_handle == 'log10':
        fcn_handle = 'log_{10}'

    if fcn_handle == 'sqrt':
        return '\\sqrt { ' + arg1 + ' }'
    else:
        return '\\' + fcn_handle + ' \\left( ' + arg1 + ' \\right) '


def handle_equality_node(node, arg1, arg2):
    return arg1 + ' = ' + arg2


def handle_inequality_node(node, arg1, arg2):
    return arg1 + ' \leq ' + arg2


def handle_scalarVar_node(node):
    return node.name


def handle_num_node(node):
    if isinstance(node, float):
        if node.is_integer():
            node = int(node)
    return str(node)


def handle_sum_expression(node, *args):
    rstr = args[0]
    for i in range(1, len(args)):
        if args[i][0] == '-':
            rstr += ' - ' + args[i][1:]
        else:
            rstr += ' + ' + args[i]
    return rstr


def handle_monomialTermExpression_node(node, arg1, arg2):
    if arg1 == '1':
        return arg2
    elif arg1 == '-1':
        return '-' + arg2
    else:
        return arg1 + arg2


def handle_named_expression_node(node, arg1):
    return arg1


def handle_ranged_inequality_node(node, arg1, arg2, arg3):
    return arg1 + ' \\leq ' + arg2 + ' \\leq ' + arg3


def handle_exprif_node(node, arg1, arg2, arg3):
    return 'f_{\\text{exprIf}}(' + arg1 + ',' + arg2 + ',' + arg3 + ')'

    ## Raises not implemented error
    # raise NotImplementedError('Expr_if objects not supported by the Latex Printer')

    ## Puts cases in a bracketed matrix
    # pstr = ''
    # pstr += '\\begin{Bmatrix} '
    # pstr += arg2 + ' , & ' + arg1 + '\\\\ '
    # pstr += arg3 + ' , & \\text{otherwise}' + '\\\\ '
    # pstr += '\\end{Bmatrix}'
    # return pstr


def handle_external_function_node(node, *args):
    pstr = ''
    pstr += 'f('
    for i in range(0, len(args) - 1):
        pstr += args[i]
        if i <= len(args) - 3:
            pstr += ','
        else:
            pstr += ')'
    return pstr


def handle_functionID_node(node, *args):
    # seems to just be a placeholder empty wrapper object
    return ''


def handle_indexedVar_node(node, *args):
    return node.name


def handle_indexTemplate_node(node, *args):
    return '__INDEX_PLACEHOLDER_8675309_GROUP_%s_%s__' % (node._group, node._set)


def handle_numericGIE_node(node, *args):
    pstr = ''
    pstr += args[0] + '_{'
    for i in range(1, len(args)):
        pstr += args[i]
        if i <= len(args) - 2:
            pstr += ','
        else:
            pstr += '}'
    return pstr


def handle_templateSumExpression_node(node, *args):
    pstr = ''
    pstr += '\\sum_{%s} %s' % (
        '__SET_PLACEHOLDER_8675309_GROUP_%s_%s__'
        % (node._iters[0][0]._group, str(node._iters[0][0]._set)),
        args[0],
    )
    return pstr


class _LatexVisitor(StreamBasedExpressionVisitor):
    def __init__(self):
        super().__init__()

        self._operator_handles = {
            ScalarVar: handle_scalarVar_node,
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
            _GeneralVarData: handle_scalarVar_node,
            ScalarObjective: handle_named_expression_node,
            kernel.objective.objective: handle_named_expression_node,
            ExternalFunctionExpression: handle_external_function_node,
            _PythonCallbackFunctionID: handle_functionID_node,
            LinearExpression: handle_sum_expression,
            SumExpression: handle_sum_expression,
            MonomialTermExpression: handle_monomialTermExpression_node,
            IndexedVar: handle_indexedVar_node,
            IndexTemplate: handle_indexTemplate_node,
            Numeric_GetItemExpression: handle_numericGIE_node,
            TemplateSumExpression: handle_templateSumExpression_node,
        }

    def exitNode(self, node, data):
        return self._operator_handles[node.__class__](node, *data)


def latex_printer(
    pyomoElement, filename=None, useAlignEnvironment=False, splitContinuousSets=False
):
    '''
    This function produces a string that can be rendered as LaTeX

    pyomoElement: The thing to be printed to LaTeX.  Accepts Blocks (including models), Constraints, and Expressions

    filename: An optional file to write the LaTeX to.  Default of None produces no file

    useAlignEnvironment: Default behavior uses equation/aligned and produces a single LaTeX Equation (ie, ==False).
        Setting this input to True will instead use the align environment, and produce equation numbers for each
        objective and constraint.  Each objective and constraint will be labeled with its name in the pyomo model.
        This flag is only relevant for Models and Blocks.

    splitContinuous: Default behavior has all sum indices be over "i \in I" or similar.  Setting this flag to
        True makes the sums go from: \sum_{i=1}^{5} if the set I is continuous and has 5 elements

    '''

    # Various setup things
    isSingle = False
    if not isinstance(pyomoElement, _BlockData):
        if isinstance(pyomoElement, pe.Objective):
            objectives = [pyomoElement]
            constraints = []
            expressions = []
            templatize_fcn = templatize_constraint

        if isinstance(pyomoElement, pe.Constraint):
            objectives = []
            constraints = [pyomoElement]
            expressions = []
            templatize_fcn = templatize_constraint

        if isinstance(pyomoElement, pe.Expression):
            objectives = []
            constraints = []
            expressions = [pyomoElement]
            templatize_fcn = templatize_expression

        if isinstance(pyomoElement, ExpressionBase):
            objectives = []
            constraints = []
            expressions = [pyomoElement]
            templatize_fcn = templatize_passthrough

        useAlignEnvironment = False
        isSingle = True
    else:
        objectives = [
            obj
            for obj in pyomoElement.component_data_objects(
                pe.Objective, descend_into=True, active=True
            )
        ]
        constraints = [
            con
            for con in pyomoElement.component_objects(
                pe.Constraint, descend_into=True, active=True
            )
        ]
        expressions = []
        templatize_fcn = templatize_constraint

    # In the case where just a single expression is passed, add this to the constraint list for printing
    constraints = constraints + expressions

    # Declare a visitor/walker
    visitor = _LatexVisitor()

    # starts building the output string
    pstr = ''
    if useAlignEnvironment:
        pstr += '\\begin{align} \n'
        tbSpc = 4
    else:
        pstr += '\\begin{equation} \n'
        if not isSingle:
            pstr += '    \\begin{aligned} \n'
            tbSpc = 8
        else:
            tbSpc = 4

    # Iterate over the objectives and print
    for obj in objectives:
        obj_template, obj_indices = templatize_fcn(obj)
        if obj.sense == 1:
            pstr += ' ' * tbSpc + '& \\text{%s} \n' % ('minimize')
        else:
            pstr += ' ' * tbSpc + '& \\text{%s} \n' % ('maximize')

        pstr += ' ' * tbSpc + '& & %s ' % (visitor.walk_expression(obj_template))
        if useAlignEnvironment:
            pstr += '\\label{obj:' + m.name + '_' + obj.name + '} '
        pstr += '\\\\ \n'

    # Iterate over the constraints
    if len(constraints) > 0:
        # only print this if printing a full formulation
        if not isSingle:
            pstr += ' ' * tbSpc + '& \\text{subject to} \n'

        # first constraint needs different alignment because of the 'subject to'
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
            con_template, indices = templatize_fcn(con)

            # Walk the constraint
            conLine = (
                ' ' * tbSpc + algn + ' %s ' % (visitor.walk_expression(con_template))
            )

            # Multiple constraints are generated using a set
            if len(indices) > 0:
                nm = indices[0]._set
                gp = indices[0]._group

                ixTag = '__INDEX_PLACEHOLDER_8675309_GROUP_%s_%s__' % (gp, nm)
                stTag = '__SET_PLACEHOLDER_8675309_GROUP_%s_%s__' % (gp, nm)

                conLine += ', \\quad %s \\in %s ' % (ixTag, stTag)
            pstr += conLine

            # Add labels as needed
            if useAlignEnvironment:
                pstr += '\\label{con:' + m.name + '_' + con.name + '} '

            # prevents an emptly blank line from being at the end of the latex output
            if i <= len(constraints) - 2:
                pstr += tail
            else:
                pstr += '\n'

    # close off the print string
    if useAlignEnvironment:
        pstr += '\\end{align} \n'
    else:
        if not isSingle:
            pstr += '    \\end{aligned} \n'
            pstr += '    \\label{%s} \n ' % (m.name)
        pstr += '\end{equation} \n'

    # Handling the iterator indices

    # preferential order for indices
    setPreferenceOrder = [
        'I',
        'J',
        'K',
        'M',
        'N',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
    ]

    # Go line by line and replace the placeholders with correct set names and index names
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
            continuousSets = dict(
                zip(uniqueSets, [False for i in range(0, len(uniqueSets))])
            )
            for i in range(0, len(uniqueSets)):
                st = getattr(m, uniqueSets[i])
                stData = st.data()
                stCont = True
                for ii in range(0, len(stData)):
                    if ii + stData[0] != stData[ii]:
                        stCont = False
                        break
                continuousSets[uniqueSets[i]] = stCont

            # Add the continuous set data to the groupMap
            for ky, vl in groupMap.items():
                groupMap[ky].append(continuousSets[vl[0]])

            # Set up new names for duplicate sets
            assignedSetNames = []
            gmk_list = list(groupMap.keys())
            for i in range(0, len(groupMap.keys())):
                ix = gmk_list[i]
                # set not already used
                if groupMap[str(ix)][0] not in assignedSetNames:
                    assignedSetNames.append(groupMap[str(ix)][0])
                    groupMap[str(ix)].append(groupMap[str(ix)][0])
                else:
                    # Pick a new set from the preference order
                    for j in range(0, len(setPreferenceOrder)):
                        stprf = setPreferenceOrder[j]
                        # must not be already used
                        if stprf not in assignedSetNames:
                            assignedSetNames.append(stprf)
                            groupMap[str(ix)].append(stprf)
                            break

            # set up the substitutions
            setStrings = {}
            indexStrings = {}
            for ky, vl in groupMap.items():
                setStrings['__SET_PLACEHOLDER_8675309_GROUP_%s_%s__' % (ky, vl[0])] = [
                    vl[2],
                    vl[1],
                    vl[0],
                ]
                indexStrings[
                    '__INDEX_PLACEHOLDER_8675309_GROUP_%s_%s__' % (ky, vl[0])
                ] = vl[2].lower()

            # replace the indices
            for ky, vl in indexStrings.items():
                ln = ln.replace(ky, vl)

            # replace the sets
            for ky, vl in setStrings.items():
                # if the set is continuous and the flag has been set
                if splitContinuousSets and vl[1]:
                    st = getattr(m, vl[2])
                    stData = st.data()
                    bgn = stData[0]
                    ed = stData[-1]
                    ln = ln.replace(
                        '\\sum_{%s}' % (ky),
                        '\\sum_{%s = %d}^{%d}' % (vl[0].lower(), bgn, ed),
                    )
                    ln = ln.replace(ky, vl[2])
                else:
                    # if the set is not continuous or the flag has not been set
                    st = getattr(m, vl[2])
                    stData = st.data()
                    bgn = stData[0]
                    ed = stData[-1]
                    ln = ln.replace(
                        '\\sum_{%s}' % (ky),
                        '\\sum_{%s \\in %s}' % (vl[0].lower(), vl[0]),
                    )
                    ln = ln.replace(ky, vl[2])

        # Assign the newly modified line
        latexLines[jj] = ln

    # rejoin the corrected lines
    pstr = '\n'.join(latexLines)

    # optional write to output file
    if filename is not None:
        fstr = ''
        fstr += '\\documentclass{article} \n'
        fstr += '\\usepackage{amsmath} \n'
        fstr += '\\begin{document} \n'
        fstr += pstr
        fstr += '\\end{document} \n'
        f = open(filename, 'w')
        f.write(fstr)
        f.close()

    # return the latex string
    return pstr
