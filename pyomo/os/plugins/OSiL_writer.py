import logging
import xml.dom

import pyutilib
from pyomo.opt.base import problem, AbstractProblemWriter, ProblemFormat
import pyomo.core

try:
    xrange = xrange
except:
    xrange = range

convert_name = lambda x: x.replace('[','(').replace(']',')')

logger = logging.getLogger('pyomo.os')

class ProblemWriter_osil(AbstractProblemWriter):
    """
    This is a writer instance for the OSiL xml specification.  See "OSiL: An
    Instance Language for Optimization" by Robert Fourer, Jun Ma, Kipp Martin.
    Likely available at:

    http://www.coin-or.org/OS/publications/ (index of papers, Jun 2010)
    """
    pyomo.misc.plugin.alias('osil')

    def __init__( self ):
        AbstractProblemWriter.__init__( self, ProblemFormat.osil )


    def __call__(self, model, filename, solver_capability):
        if filename is None:
            filename = model.name + '.xml'
        #
        output = open( filename, 'w' )
        self._print_model( model, output, solver_capability )
        output.close()
        #
        return filename, None


    def _print_model( self, model, output, solver_capability ):
        dom = xml.dom.getDOMImplementation()
        doc = dom.createDocument(None, None, None)
        osil = doc.createElement('osil')
        osil.appendChild( self._getInstanceHeader( doc, model ) )
        osil.appendChild( self._getInstanceData( doc, model ) )
        doc.appendChild( osil )
        #
        output.write( doc.toprettyxml(indent="  ") )


    def _getInstanceHeader ( self, doc, model ):
        header = doc.createElement('instanceHeader')
        #
        name   = doc.createElement('name')
        name.appendChild( doc.createTextNode( model.name ) )
        header.appendChild( name )
        #
        #source = doc.createElement('source')
        #source.appendChild( doc.createTextNode( model.name ) )
        #header.appendChild( source )
        #
        description = doc.createElement('description')
        description.appendChild( doc.createTextNode( "Created OSiL problem from Pyomo model '%s'" % model.name ) )
        header.appendChild( description )
        #
        return header


    def _getInstanceData ( self, doc, model ):
        nodeFunctions = (
            self._getVariablesElement,
            self._getObjectivesElement,
            self._getConstraintsElement,
            self._getLinearConstraintCoefficientsElement,
            self._getQuadraticCoefficientsElement,
            #self._getNonLinearExpressionsElement,
            #self._getSpecialOrderedSetsElement,
        )
        #
        instanceData = doc.createElement('instanceData')
        for f in nodeFunctions:
            instanceData.appendChild( f(doc, model) )
        return instanceData


    def _getSpecialOrderedSetsElement ( self, doc, model ):
        return None


    def _getNonLinearExpressionsElement ( self, doc, model ):
        nle = doc.createElement('nonlinearExpressions')

        def create_sub_expression ( binary_action, nodes ):
            """
            Creates a sub expression of terms that need a similar action.

            For example, create_sub_expression('times', [el1, el2, el3, el4] )
            would return:

            <times><el1><times><el2><times><el3><el4></times></times></times>
            """
            if len(nodes) > 1: # multiple vars in this term
                node = doc.createElement( binary_action )
                node.appendChild( nodes.pop() )
                node.appendChild( nodes.pop() )
                for var in nodes:
                    sub_node = doc.createElement( binary_action )
                    sub_node.appendChild( node )
                    sub_node.appendChild( var )
                    node = sub_node
                expression = node
            elif len(nodes) == 1:
                expression = nodes.pop()
            else:
                # TODO include a better error message
                raise Exception('Pyomo internal programming error.\n'        \
                'Please inform developers of what you did to encounter this ' \
                'message: Writer, Nonlinear Sub expression creator')

            return expression


        def _create_nl_expression ( expression, index ):
            if pyomo.core.is_quadratic( expression ) or not pyomo.core.is_nonlinear( expression ):
                return None

            order_nodes = list()
            for order in sorted( expression.keys() ):
                if order in (-1, 0, 1, 2):
                    continue
                try:
                    dummy = int(order)
                except:
                    msg  = 'Unable to write OSiL format of model.\n'          \
                    'Explanation: The Pyomo OSiL Writer has not implemented ' \
                    'a nonlinear expression type that you have in your '      \
                    'objective or constraints.  (For example, sin(x), ln(x),' \
                    ' or x^(2.3).)'
                    raise Exception(msg)

                order = expression[order] # get the actual expression part
                term_nodes = list()
                for term in order:
                    coef = order[term]

                    var_nodes = list()
                    for var in term:
                        vname = convert_name( expression[-1][var].name )
                        vid   = expression[-1][var].id

                        var_node = doc.createElement('variable')
                        var_node.setAttribute('name', vname )
                        var_node.setAttribute('idx', str(vid) )

                        power = term[var]
                        if power > 1:
                            raised_to = doc.createElement('number')
                            raised_to.setAttribute('value', str(power) )
                            power_node = doc.createElement('power')
                            power_node.appendChild( var_node )
                            power_node.appendChild( raised_to )
                            var_node = power_node

                        var_nodes.append( var_node )

                    sub_expr = create_sub_expression('times', var_nodes )

                    term_node = doc.createElement('times')
                    coef_node = doc.createElement('number')
                    coef_node.setAttribute('value', str(coef) )
                    term_node.appendChild( coef_node )
                    term_node.appendChild( sub_expr )

                    term_nodes.append( term_node )

                sub_expr = create_sub_expression('plus', term_nodes )
                order_nodes.append( sub_expr )

            sub_expr = create_sub_expression('plus', order_nodes )

            nl = doc.createElement('nl')
            nl.setAttribute('idx', str(index) )
            nl.appendChild( sub_expr )

            return nl


        def create_nl_expressions ( expressions_set, **kwargs ):
            index    = kwargs.pop('index', 0 )   # expression index
            modifier = kwargs.pop('modifer', 1 ) # next expression index?
            for es in expressions_set:
                for key in expressions_set[es]:
                    expression = expressions_set[es][key].repn
                    nlexpression = _create_nl_expression( expression, index )
                    if nlexpression:
                        nle.appendChild( nlexpression )

                    index += modifier

        objectives  = model.active_components( pyomo.core.Objective )
        constraints = model.active_components( pyomo.core.Constraint )
        create_nl_expressions( objectives, index=-1, modifier=-1  )
        create_nl_expressions( constraints )

        nle.setAttribute('number', str(len( nle.childNodes )) )

        return nle


    def _getLinearConstraintCoefficientsElement ( self, doc, model ):
        lcc = doc.createElement('linearConstraintCoefficients')


        # TODO: figure out when to use colIdx
        start_node = doc.createElement('start')
        col_node   = doc.createElement('colIdx')
        value_node = doc.createElement('value')

        def get_linear_info ( expression ):
            terms = []
            if 1 in expression:
                for term_id in expression[1]:
                    coef = expression[1][term_id]
                    terms.append( (term_id, coef) )
            return terms

        start = 0 # Constraint count.  OSiL orders the constraints, hence the
                # sorted version of the active_components, above
        constraints = model.active_components( pyomo.core.Constraint )
        for con in constraints:
            for index in sorted(constraints[con]):
                C = constraints[con][index]
                num = 0
                el_beg = doc.createElement('el')
                for col, value in get_linear_info( C.repn ):
                    num += 1
                    el_val = doc.createElement('el')
                    el_col = doc.createElement('el')
                    el_val.appendChild( doc.createTextNode( str(value)) )
                    el_col.appendChild( doc.createTextNode( str(col)) )
                    value_node.appendChild( el_val )
                    col_node.appendChild( el_col )
                el_beg.appendChild( doc.createTextNode( str(start)) )
                start_node.appendChild( el_beg )
                start += num
        lcc.appendChild( start_node )
        lcc.appendChild( col_node )
        lcc.appendChild( value_node )

        lcc.setAttribute('numberOfValues', str(len(col_node.childNodes)) )

        return lcc


    def _getConstraintsElement ( self, doc, model ):
        constraint = doc.createElement('constraints')

        def get_bound ( expression, offset=0.0 ):
            if expression.is_constant():
                return expression() + offset
            else:
                msg  = 'Non-constant constraint bound.  Expression type: '
                msg += expression.__class__.__name__
                # TODO: Convert exp.pprint to /return/ a string, not directly
                # print.  Then it's usable in any context, like below
                # if exp is not None:
                #     msg += exp.pprint()
                raise ValueError(msg)

        constraints = model.active_components( pyomo.core.Constraint )

        for con in constraints:
            for index in sorted(constraints[con]):
                node = doc.createElement('con')
                C = constraints[con][index]

                if index is None:
                    name = convert_name( C.name )
                else:
                    name = convert_name( C.name+'(%s)' % str(index) )
                if 0 in C.repn:
                    offset = C.repn[0][None]
                else:
                    offset = 0.0

                node.setAttribute('name', name)

                if C._equality: # is an equality constraint
                    node.setAttribute('constant', str( get_bound(C.lower, -offset) ))
                else: # is an inequality constraint
                    if C.lower is not None:
                        node.setAttribute('lb', str( get_bound(C.lower, -offset) ))
                    if C.upper is not None:
                        node.setAttribute('ub', str( get_bound(C.upper, -offset) ))

                constraint.appendChild( node )

        constraint.setAttribute('numberOfConstraints', str(len(constraint.childNodes)) )

        return constraint


    def _getQuadraticCoefficientsElement ( self, doc, model ):
        quadratic = doc.createElement('quadraticCoefficients')

        qterm_index_attributes = ('idxOne', 'idxTwo')

        def add_qterms ( objs, qidx=True ):
            for idx in xrange(len(objs)):
                qterm_idx = str(idx)
                if qidx != True:
                    qterm_idx = str(-1) # -1 for objectives

                obj = objs[idx]

                for key in obj:
                    expression = obj[key].repn
                    if pyomo.core.is_constant( expression ):
                        # we've already informed the user we're ignoring this
                        # Object, so ignore it and move on
                        continue

                    if 2 in expression: # quadratic terms
                        keys = expression[2].keys()
                        for quadterm in tuple(keys):
                            qterm = doc.createElement('qterm')
                            i = iter( qterm_index_attributes )

                            for var_id in quadterm:
                                var_index = str( var_id[1] )
                                count = quadterm[var_id]

                                if ( 2 == count ):  # same var (e.g. X**2)
                                    qterm.setAttribute( i.next(), var_index )
                                    qterm.setAttribute( i.next(), var_index )
                                else: # different vars (e.g. X*Y)
                                    qterm.setAttribute( i.next(), var_index )

                            coef = str( expression[2][quadterm] )
                            qterm.setAttribute('coef', coef )
                            qterm.setAttribute('idx', qterm_idx )
                            quadratic.appendChild( qterm )

        # The quadratics section of the XML document deals with both
        # objectives and constraints, so we add them in two parts.
        objectives = model.active_components( pyomo.core.Objective )
        constraints = model.active_components( pyomo.core.Constraint )
        #add_qterms( objectives, False )
        #add_qterms( constraints )

        quadratic.setAttribute('numberOfQuadraticTerms', str(len(quadratic.childNodes)) )

        return quadratic


    def _getObjectivesElement ( self, doc, model ):
        #
        # don't worry about non-linear parts here.  It's checked as
        # part of _getNonlinearExpressions
        #
        objectives = doc.createElement('objectives')

        objs = model.active_components( pyomo.core.Objective )


        for objarray in objs:
            if objs[objarray].is_minimizing():
                sense = 'min'
            else:
                sense = 'max'

            for key in objs[objarray]: # note: None is a valid dict key
                obj = objs[objarray][key]
                expression = obj.repn
                if pyomo.core.is_constant( expression ):
                    msg = "Ignoring objective '%s[%s]' which is constant"
                    logger.warning( msg % (str(objarray), str(key)) )
                    continue
                #
                obj_node = doc.createElement('obj')
                #
                print(expression)
                if 1 in expression: # first-order terms
                    keys = sorted( expression[1].keys() )
                    for var_key in keys:
                        var_index = expression[-1][var_key].id
                        coef = expression[1][var_key]
                        value = doc.createTextNode( str(coef) )

                        coef_node = doc.createElement('coef')
                        coef_node.setAttribute('idx', str(var_index) )
                        coef_node.appendChild( value )

                        obj_node.appendChild( coef_node )

                obj_node.setAttribute('maxOrMin', sense )
                if key is None:
                    obj_node.setAttribute('name', objs[objarray].name)
                else:
                    obj_node.setAttribute('name', objs[objarray].name+'('+str(key)+')')
                obj_node.setAttribute('numberOfObjCoef', str(len(obj_node.childNodes)) )

                objectives.appendChild( obj_node )


        objectives.setAttribute('numberOfObjectives', str(len(objectives.childNodes)) )

        return objectives


    def _getVariablesElement ( self, doc, model ):
        variables = doc.createElement('variables')

        vars = model.active_components( pyomo.core.Var )

        for vararray in vars:
            vtable = {}
            for key in vars[vararray]:
                var = vars[vararray][key]

                if key is None:
                    name = '%s' % str(vararray)
                else:
                    name = '%s(%s)' % (str(vararray), str(key))
                name = ''.join(name.split()) # remove whitespace
                name = name.replace('((', '(').replace('))', ')')

                ubound = 'INF'
                lbound = '-INF'
                if not var.ub is None: ubound = str(var.ub.value)
                if not var.lb is None: lbound = str(var.lb.value)

                if   ( var.is_continuous() ): vtype = 'C'   # continuous
                elif ( var.is_binary() ):     vtype = 'B'   # binary
                elif ( var.is_integer() ):    vtype = 'I'   # integer
                else:                         vtype = 'S'   # string
                # TODO: can Pyomo represent 'D' (Semi-continuous) variables?
                # TODO: can Pyomo represent 'J' (Semi-integer) variables?

                node = doc.createElement('var')
                node.setAttribute('name', name )
                node.setAttribute('lb', lbound )
                node.setAttribute('ub', ubound )
                node.setAttribute('type', vtype )

                vtable[var.id] = node
            for key in sorted(vtable.keys()):
                variables.appendChild( vtable[key] )

        variables.setAttribute('numberOfVariables', str(len(variables.childNodes)) )

        return variables

