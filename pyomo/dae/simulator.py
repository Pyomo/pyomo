#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
from pyomo.core.base import Constraint, Param, value, Suffix, Block

from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error

from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import NumericValue, native_numeric_types
from pyomo.core.base.template_expr import IndexTemplate, _GetItemIndexer

from six import iterkeys, itervalues

import logging

__all__ = ('Simulator', )
logger = logging.getLogger('pyomo.core')

# Check numpy availability
numpy_available = True
try:
    import numpy as np
except ImportError:
    numpy_available = True

# Check integrator availability
scipy_available = True
try:
    import platform
    if platform.python_implementation() == "PyPy":  # pragma:nocover
        # scipy is importable into PyPy, but ODE integrators don't work. (2/18)
        raise ImportError
    import scipy.integrate as scipy
except ImportError:
    scipy_available = False

casadi_available = True
try:
    import casadi
    casadi_intrinsic = {
            'log': casadi.log,
            'log10': casadi.log10,
            'sin': casadi.sin,
            'cos': casadi.cos,
            'tan': casadi.tan,
            'cosh': casadi.cosh,
            'sinh': casadi.sinh,
            'tanh': casadi.tanh,
            'asin': casadi.asin,
            'acos': casadi.acos,
            'atan': casadi.atan,
            'exp': casadi.exp,
            'sqrt': casadi.sqrt,
            'asinh': casadi.asinh,
            'acosh': casadi.acosh,
            'atanh': casadi.atanh,
            'ceil': casadi.ceil,
            'floor': casadi.floor}
except ImportError:
    casadi_available = False


def _check_getitemexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    GetItemExpression at expr.arg(i) to see if it is a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the :py:class:`DerivativeVar<DerivativeVar>` and
    the RHS. If not, return None.
    """
    if type(expr.arg(i)._base) is DerivativeVar:
        return [expr.arg(i), expr.arg(1 - i)]
    else:
        return None


def _check_productexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    ProductExpression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    expr_ = expr.arg(i)
    stack = [(expr_, 1)]
    pterms = []
    dv = None
    
    while stack:
        curr, e_ = stack.pop()
        if curr.__class__ is EXPR.ProductExpression:
            stack.append((curr.arg(0), e_))
            stack.append((curr.arg(1), e_))
        elif curr.__class__ is EXPR.ReciprocalExpression:
            stack.append((curr.arg(0), - e_))
        elif type(curr) is EXPR.GetItemExpression and \
             type(curr._base) is DerivativeVar:
            dv = (curr, e_)
        else:
            pterms.append((curr, e_))

    if dv is None:
        return None

    numer = 1
    denom = 1
    for term, e_ in pterms:
        if e_ == 1:
            denom *= term 
        else:
            numer *= term 
    curr, e_ = dv
    if e_ == 1:
        return [curr, expr.arg(1 - i) * numer / denom]
    else:
        return [curr, denom / (expr.arg(1 - i) * numer)]


def _check_negationexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    NegationExpression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    arg = expr.arg(i).arg(0)

    if type(arg) is EXPR.GetItemExpression and \
       type(arg._base) is DerivativeVar:
        return [arg, - expr.arg(1 - i)]

    if type(arg) is EXPR.ProductExpression:
        lhs = arg.arg(0)
        rhs = arg.arg(1)

        if not (type(lhs) in native_numeric_types or
                    not lhs.is_potentially_variable()):
            return None
        if not (type(rhs) is EXPR.GetItemExpression and
                        type(rhs._base) is DerivativeVar):
            return None

        return [rhs, - expr.arg(1 - i) / lhs]

    return None


def _check_viewsumexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    Sum expression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    # Get the side of the equality expression with the derivative variable
    sumexp = expr.arg(i)
    items = []
    dv = None
    dvcoef = 1

    for idx, item in enumerate(sumexp.args):
        if dv is not None:
            items.append(item)
        elif type(item) is EXPR.GetItemExpression and \
           type(item._base) is DerivativeVar:
            dv = item
        elif type(item) is EXPR.ProductExpression:
            # This will contain the constant coefficient if there is one
            lhs = item.arg(0)
            # This is a potentially variable expression
            rhs = item.arg(1)
            if (type(lhs) in native_numeric_types or
                    not lhs.is_potentially_variable()) \
                and (type(rhs) is EXPR.GetItemExpression and
                             type(rhs._base) is DerivativeVar):
                dv = rhs
                dvcoef = lhs
        else:
            items.append(item)

    if dv is not None:
        # Form the "other" side of the equality expression
        RHS = expr.arg(1 - i)
        for item in items:
            RHS -= item
        RHS = RHS / dvcoef
        return [dv, RHS]

    return None


if scipy_available:
    class Pyomo2Scipy_Visitor(EXPR.ExpressionReplacementVisitor):
        """
        Expression walker that replaces _GetItemExpression
        instances with mutable parameters.
        """

        def __init__(self, templatemap):
            super(Pyomo2Scipy_Visitor, self).__init__()
            self.templatemap = templatemap

        def visiting_potential_leaf(self, node):
            if type(node) is IndexTemplate:
                return True, node

            if type(node) is EXPR.GetItemExpression:
                _id = _GetItemIndexer(node)
                if _id not in self.templatemap:
                    self.templatemap[_id] = Param(mutable=True)
                    self.templatemap[_id].construct()
                    _args = []
                    self.templatemap[_id]._name = "%s[%s]" % (
                        node._base.name, ','.join(str(x) for x in _id._args))
                return True, self.templatemap[_id]

            return super(
                Pyomo2Scipy_Visitor, self).visiting_potential_leaf(node)


def convert_pyomo2scipy(expr, templatemap):
    """Substitute _GetItem nodes in an expression tree.

    This substition function is used to replace Pyomo _GetItem
    nodes with mutable Params.

    Args:
        templatemap: dictionary mapping _GetItemIndexer objects to
            mutable params

    Returns:
        a new expression tree with all substitutions done
    """
    if not scipy_available:
        raise DAE_Error("SciPy is not installed. Cannot substitute SciPy "
                        "intrinsic functions.")
    visitor = Pyomo2Scipy_Visitor(templatemap)
    return visitor.dfs_postorder_stack(expr)


if casadi_available:
    class Substitute_Pyomo2Casadi_Visitor(EXPR.ExpressionReplacementVisitor):
        """
        Expression walker that replaces 

           * _UnaryFunctionExpression instances with unary functions that
             point to casadi intrinsic functions.

           * _GetItemExpressions with _GetItemIndexer objects that references
             CasADi variables.
        """

        def __init__(self, templatemap):
            super(Substitute_Pyomo2Casadi_Visitor, self).__init__()
            self.templatemap = templatemap

        def visit(self, node, values):
            """Replace a node if it's a unary function."""
            if type(node) is EXPR.UnaryFunctionExpression:
                return EXPR.UnaryFunctionExpression(
                                values[0], 
                                node._name, 
                                casadi_intrinsic[node._name])
            return node

        def visiting_potential_leaf(self, node):
            """Replace a node if it's a _GetItemExpression."""
            if type(node) is EXPR.GetItemExpression:
                _id = _GetItemIndexer(node)
                if _id not in self.templatemap:
                    name = "%s[%s]" % (
                        node._base.name, ','.join(str(x) for x in _id._args))
                    self.templatemap[_id] = casadi.SX.sym(name)
                return True, self.templatemap[_id]

            if type(node) in native_numeric_types or \
               not node.is_expression_type() or \
               type(node) is IndexTemplate:
                return True, node

            return False, None


    class Convert_Pyomo2Casadi_Visitor(EXPR.ExpressionValueVisitor):
        """
        Expression walker that evaluates an expression 
        generated by the Substitute_Pyomo2Casadi_Visitor walker.

        In Coopr3 this walker was not necessary because the expression could
        be simply evaluated.  But in Pyomo5, the evaluation logic was
        changed to be non-recursive, which involves checks on the types of
        leaves in the expression tree. Hence, the evaluation logic fails if
        leaves in the tree are not standard Pyomo5 variable types.
        """

        def visit(self, node, values):
            """ Visit nodes that have been expanded """
            return node._apply_operation(values)

        def visiting_potential_leaf(self, node):
            """ 
            Visiting a potential leaf.

            Return True if the node is not expanded.
            """
            if node.__class__ in native_numeric_types:
                return True, node

            if node.__class__ is casadi.SX:
                return True, node

            if node.is_variable_type():
                return True, value(node)

            if not node.is_expression_type():
                return True, value(node)

            return False, None


def substitute_pyomo2casadi(expr, templatemap):
    """Substitute IndexTemplates in an expression tree.

    This substition function is used to replace Pyomo intrinsic
    functions with CasADi functions.

    Args:
        expr: a Pyomo expression
        templatemap: dictionary mapping _GetItemIndexer objects to
            mutable params

    Returns:
        a new expression tree with all substitutions done
    """
    if not casadi_available:
        raise DAE_Error("CASADI is not installed.  Cannot substitute CasADi "
                        "variables and intrinsic functions.")
    visitor = Substitute_Pyomo2Casadi_Visitor(templatemap)
    return visitor.dfs_postorder_stack(expr)


def convert_pyomo2casadi(expr):
    """Convert a Pyomo expression tree to Casadi.

    This function replaces a Pyomo expression with a CasADi expression.
    This assumes that the `substitute_pyomo2casadi` function has
    been called, so the Pyomo expression contains CasADi variables
    and intrinsic functions.  The resulting expression can be used
    with the CasADi integrator.

    Args:
        expr: a Pyomo expression with CasADi variables and intrinsic
            functions

    Returns:
        a CasADi expression tree.
    """
    if not casadi_available:
        raise DAE_Error("CASADI is not installed.  Cannot convert a Pyomo "
                        "expression to a Casadi expression.")
    visitor = Convert_Pyomo2Casadi_Visitor()
    return visitor.dfs_postorder_stack(expr)


class Simulator:
    """
    Simulator objects allow a user to simulate a dynamic model formulated
    using pyomo.dae.

    Parameters
    ----------
    m : Pyomo Model
        The Pyomo model to be simulated should be passed as the first argument

    package : `string`
        The Python simulator package to use. Currently 'scipy' and 'casadi' are
        the only supported packages
    """

    def __init__(self, m, package='scipy'):
        
        self._intpackage = package
        if self._intpackage not in ['scipy', 'casadi']:
            raise DAE_Error(
                "Unrecognized simulator package %s. Please select from "
                "%s" % (self._intpackage, ['scipy', 'casadi']))

        if self._intpackage == 'scipy':
            if not scipy_available:
                # Converting this to a warning so that Simulator initialization
                # can be tested even when scipy is unavailable
                logger.warning("The scipy module is not available. You may "
                               "build the Simulator object but you will not "
                               "be able to run the simulation.")
        else:
            if not casadi_available:
                # Initializing the simulator for use with casadi requires
                # access to casadi objects. Therefore, we must throw an error
                # here instead of a warning. 
                raise ValueError("The casadi module is not available. "
                                  "Cannot simulate model.")

        # Check for active Blocks and throw error if any are found
        if len(list(m.component_data_objects(Block, active=True,
                                             descend_into=False))):
            raise DAE_Error("The Simulator cannot handle hierarchical models "
                            "at the moment.")

        temp = m.component_map(ContinuousSet)
        if len(temp) != 1:
            raise DAE_Error(
                "Currently the simulator may only be applied to "
                "Pyomo models with a single ContinuousSet")

        # Get the ContinuousSet in the model
        contset = list(temp.values())[0]

        # Create a index template for the continuous set
        cstemplate = IndexTemplate(contset)

        # Ensure that there is at least one derivative in the model
        derivs = m.component_map(DerivativeVar)
        derivs = list(derivs.keys())

        if hasattr(m, '_pyomo_dae_reclassified_derivativevars'):
            for d in m._pyomo_dae_reclassified_derivativevars:
                derivs.append(d.name)
        if len(derivs) == 0:
            raise DAE_Error("Cannot simulate a model with no derivatives")

        templatemap = {}  # Map for template substituter
        rhsdict = {}  # Map of derivative to its RHS templated expr
        derivlist = []  # Ordered list of derivatives
        alglist = []  # list of templated algebraic equations

        # Loop over constraints to find differential equations with separable
        # RHS. Must find a RHS for every derivative var otherwise ERROR. Build
        # dictionary of DerivativeVar:RHS equation.
        for con in m.component_objects(Constraint, active=True):
            
            # Skip the discretization equations if model is discretized
            if '_disc_eq' in con.name:
                continue
            
            # Check dimension of the Constraint. Check if the
            # Constraint is indexed by the continuous set and
            # determine its order in the indexing sets
            if con.dim() == 0:
                continue
            elif con._implicit_subsets is None:
                # Check if the continuous set is the indexing set
                if con._index is not contset:
                    continue
                else:
                    csidx = 0
                    noncsidx = (None,)
            else:
                temp = con._implicit_subsets
                dimsum = 0
                csidx = -1
                noncsidx = None
                for s in temp:
                    if s is contset:
                        if csidx != -1:
                            raise DAE_Error(
                                "Cannot simulate the constraint %s because "
                                "it is indexed by duplicate ContinuousSets"
                                % con.name)
                        csidx = dimsum
                    elif noncsidx is None:
                        noncsidx = s
                    else:
                        noncsidx = noncsidx.cross(s)
                    dimsum += s.dimen
                if csidx == -1:
                    continue

            # Get the rule used to construct the constraint
            conrule = con.rule

            for i in noncsidx:
                # Insert the index template and call the rule to
                # create a templated expression              
                if i is None:
                    tempexp = conrule(m, cstemplate)
                else:
                    if not isinstance(i, tuple):
                        i = (i,)
                    tempidx = i[0:csidx] + (cstemplate,) + i[csidx:]
                    tempexp = conrule(m, *tempidx)

                # Check to make sure it's an EqualityExpression
                if not type(tempexp) is EXPR.EqualityExpression:
                    continue
            
                # Check to make sure it's a differential equation with
                # separable RHS
                args = None
                # Case 1: m.dxdt[t] = RHS 
                if type(tempexp.arg(0)) is EXPR.GetItemExpression:
                    args = _check_getitemexpression(tempexp, 0)
            
                # Case 2: RHS = m.dxdt[t]
                if args is None:
                    if type(tempexp.arg(1)) is EXPR.GetItemExpression:
                        args = _check_getitemexpression(tempexp, 1)

                # Case 3: m.p*m.dxdt[t] = RHS
                if args is None:
                    if type(tempexp.arg(0)) is EXPR.ProductExpression or \
                       type(tempexp.arg(0)) is EXPR.ReciprocalExpression:
                        args = _check_productexpression(tempexp, 0)

                # Case 4: RHS =  m.p*m.dxdt[t]
                if args is None:
                    if type(tempexp.arg(1)) is EXPR.ProductExpression or \
                       type(tempexp.arg(1)) is EXPR.ReciprocalExpression:
                        args = _check_productexpression(tempexp, 1)

                # Case 5: m.dxdt[t] + sum(ELSE) = RHS
                # or CONSTANT + m.dxdt[t] = RHS
                if args is None:
                    if type(tempexp.arg(0)) is EXPR.SumExpression:
                        args = _check_viewsumexpression(tempexp, 0)

                # Case 6: RHS = m.dxdt[t] + sum(ELSE)
                if args is None:
                    if type(tempexp.arg(1)) is EXPR.SumExpression:
                        args = _check_viewsumexpression(tempexp, 1)

                # Case 7: RHS = m.p*m.dxdt[t] + CONSTANT
                # This case will be caught by Case 6 if p is immutable. If
                # p is mutable then this case will not be detected as a
                # separable differential equation

                # Case 8: - dxdt[t] = RHS
                if args is None:
                    if type(tempexp.arg(0)) is EXPR.NegationExpression:
                        args = _check_negationexpression(tempexp, 0)

                # Case 9: RHS = - dxdt[t]
                if args is None:
                    if type(tempexp.arg(1)) is EXPR.NegationExpression:
                        args = _check_negationexpression(tempexp, 1)

                # At this point if args is not None then args[0] contains
                # the _GetItemExpression for the DerivativeVar and args[1]
                # contains the RHS expression. If args is None then the
                # constraint is considered an algebraic equation
                if args is None:
                    # Constraint is an algebraic equation or unsupported
                    # differential equation
                    if self._intpackage == 'scipy':
                        raise DAE_Error(
                            "Model contains an algebraic equation or "
                            "unrecognized differential equation. Constraint "
                            "'%s' cannot be simulated using Scipy. If you are "
                            "trying to simulate a DAE model you must use "
                            "CasADi as the integration package."
                            % str(con.name))
                    tempexp = tempexp.arg(0) - tempexp.arg(1)
                    algexp = substitute_pyomo2casadi(tempexp, templatemap)
                    alglist.append(algexp)
                    continue
            
                # Add the differential equation to rhsdict and derivlist
                dv = args[0]
                RHS = args[1]
                dvkey = _GetItemIndexer(dv)
                if dvkey in rhsdict.keys():
                    raise DAE_Error(
                        "Found multiple RHS expressions for the "
                        "DerivativeVar %s" % str(dvkey))
            
                derivlist.append(dvkey)
                if self._intpackage is 'casadi':
                    rhsdict[dvkey] = substitute_pyomo2casadi(RHS, templatemap)
                else:
                    rhsdict[dvkey] = convert_pyomo2scipy(RHS, templatemap)
        # Check to see if we found a RHS for every DerivativeVar in
        # the model
        # FIXME: Not sure how to rework this for multi-index case
        # allderivs = derivs.keys()
        # if set(allderivs) != set(derivlist):
        #     missing = list(set(allderivs)-set(derivlist))
        #     print("WARNING: Could not find a RHS expression for the "
        #     "following DerivativeVar components "+str(missing))

        # Create ordered list of differential variables corresponding
        # to the list of derivatives.
        diffvars = []

        for deriv in derivlist:
            sv = deriv._base.get_state_var()
            diffvars.append(_GetItemIndexer(sv[deriv._args]))

        # Create ordered list of algebraic variables and time-varying
        # parameters
        algvars = []

        for item in iterkeys(templatemap):
            if item._base.name in derivs:
                # Make sure there are no DerivativeVars in the
                # template map
                raise DAE_Error(
                    "Cannot simulate a differential equation with "
                    "multiple DerivativeVars")
            if item not in diffvars:
                # Finds time varying parameters and algebraic vars
                algvars.append(item)
                
        if self._intpackage == 'scipy':
            # Function sent to scipy integrator
            def _rhsfun(t, x):
                residual = []
                cstemplate.set_value(t)
                for idx, v in enumerate(diffvars):
                    if v in templatemap:
                        templatemap[v].set_value(x[idx])

                for d in derivlist:
                    residual.append(rhsdict[d]())

                return residual
            self._rhsfun = _rhsfun   
            
        # Add any diffvars not added by expression walker to self._templatemap
        if self._intpackage == 'casadi':
            for _id in diffvars:
                if _id not in templatemap:
                    name = "%s[%s]" % (
                        _id._base.name, ','.join(str(x) for x in _id._args))
                    templatemap[_id] = casadi.SX.sym(name)

        self._contset = contset
        self._cstemplate = cstemplate
        self._diffvars = diffvars
        self._derivlist = derivlist
        self._templatemap = templatemap
        self._rhsdict = rhsdict
        self._alglist = alglist
        self._algvars = algvars
        self._model = m
        self._tsim = None
        self._simsolution = None
        # The algebraic vars in the most recent simulation
        self._simalgvars = None
        # The time-varying inputs in the most recent simulation
        self._siminputvars = None

    def get_variable_order(self, vartype=None):
        """
        This function returns the ordered list of differential variable
        names. The order corresponds to the order being sent to the
        integrator function. Knowing the order allows users to provide
        initial conditions for the differential equations using a
        list or map the profiles returned by the simulate function to
        the Pyomo variables.

        Parameters
        ----------
        vartype : `string` or None
            Optional argument for specifying the type of variables to return
            the order for. The default behavior is to return the order of
            the differential variables. 'time-varying' will return the order
            of all the time-dependent algebraic variables identified in the
            model. 'algebraic' will return the order of algebraic variables
            used in the most recent call to the simulate function. 'input'
            will return the order of the time-dependent algebraic variables
            that were treated as inputs in the most recent call to the
            simulate function.

        Returns
        -------
        `list`

        """        
        if vartype == 'time-varying':
            return self._algvars
        elif vartype == 'algebraic':
            return self._simalgvars
        elif vartype == 'input':
            return self._siminputvars
        else:
            return self._diffvars
        
    def simulate(self, numpoints=None, tstep=None, integrator=None,
                 varying_inputs=None, initcon=None, integrator_options=None):
        """
        Simulate the model. Integrator-specific options may be specified as
        keyword arguments and will be passed on to the integrator.

        Parameters
        ----------
        numpoints : int
            The number of points for the profiles returned by the simulator.
            Default is 100

        tstep : int or float
            The time step to use in the profiles returned by the simulator.
            This is not the time step used internally by the integrators.
            This is an optional parameter that may be specified in place of
            'numpoints'.

        integrator : string
            The string name of the integrator to use for simulation. The
            default is 'lsoda' when using Scipy and 'idas' when using CasADi

        varying_inputs : ``pyomo.environ.Suffix``
            A :py:class:`Suffix<pyomo.environ.Suffix>` object containing the
            piecewise constant profiles to be used for certain time-varying
            algebraic variables.

        initcon : list of floats
            The initial conditions for the the differential variables. This
            is an optional argument. If not specified then the simulator
            will use the current value of the differential variables at the
            lower bound of the ContinuousSet for the initial condition.

        integrator_options : dict
            Dictionary containing options that should be passed to the
            integrator. See the documentation for a specific integrator for a
            list of valid options.

        Returns
        -------
        numpy array, numpy array
            The first return value is a 1D array of time points corresponding
            to the second return value which is a 2D array of the profiles for
            the simulated differential and algebraic variables.
        """

        if not numpy_available:
            raise ValueError("The numpy module is not available. "
                              "Cannot simulate the model.")

        if integrator_options is None:
            integrator_options = {}

        if self._intpackage == 'scipy':
            # Specify the scipy integrator to use for simulation
            valid_integrators = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
            if integrator is None:
                integrator = 'lsoda'
            elif integrator is 'odeint':
                integrator = 'lsoda'
        else:
            # Specify the casadi integrator to use for simulation.
            # Only a subset of these integrators may be used for 
            # DAE simulation. We defer this check to CasADi.
            valid_integrators = ['cvodes', 'idas', 'collocation', 'rk']
            if integrator is None:
                integrator = 'idas'

        if integrator not in valid_integrators:
            raise DAE_Error("Unrecognized %s integrator \'%s\'. Please select"
                            " an integrator from %s" % (self._intpackage,
                                                        integrator,
                                                        valid_integrators))

        # Set the time step or the number of points for the lists
        # returned by the integrator
        if tstep is not None and \
           tstep > (self._contset.last() - self._contset.first()):
            raise ValueError(
                "The step size %6.2f is larger than the span of the "
                "ContinuousSet %s" % (tstep, self._contset.name()))
        
        if tstep is not None and numpoints is not None:
            raise ValueError(
                "Cannot specify both the step size and the number of "
                "points for the simulator")
        if tstep is None and numpoints is None:
            # Use 100 points by default
            numpoints = 100

        if tstep is None:
            tsim = np.linspace(
                self._contset.first(), self._contset.last(), num=numpoints)

            # Consider adding an option for log spaced time points. Can be
            # important for simulating stiff systems.
            # tsim = np.logspace(-4,6, num=100)
            # np.log10(self._contset.first()),np.log10(
            # self._contset.last()),num=1000, endpoint=True)

        else:
            tsim = np.arange(
                self._contset.first(), self._contset.last(), tstep)

        switchpts = []
        self._siminputvars = {}
        self._simalgvars = []
        if varying_inputs is not None:
            if type(varying_inputs) is not Suffix:
                raise TypeError(
                    "Varying input values must be specified using a "
                    "Suffix. Please refer to the simulator documentation.")

            for alg in self._algvars:
                if alg._base in varying_inputs:
                    # Find all the switching points         
                    switchpts += varying_inputs[alg._base].keys()
                    # Add to dictionary of siminputvars
                    self._siminputvars[alg._base] = alg
                else:
                    self._simalgvars.append(alg)

            if self._intpackage is 'scipy' and len(self._simalgvars) != 0:
                raise DAE_Error("When simulating with Scipy you must "
                                "provide values for all parameters "
                                "and algebraic variables that are indexed "
                                "by the ContinuoutSet using the "
                                "'varying_inputs' keyword argument. "
                                "Please refer to the simulator documentation "
                                "for more information.")

            # Get the set of unique points
            switchpts = list(set(switchpts)) 
            switchpts.sort()

            # Make sure all the switchpts are within the bounds of
            # the ContinuousSet
            if switchpts[0] < self._contset.first() or \
                            switchpts[-1] > self._contset.last():
                raise ValueError("Found a switching point for one or more of "
                                 "the time-varying inputs that is not within "
                                 "the bounds of the ContinuousSet.")

            # Update tsim to include input switching points
            # This numpy function returns the unique, sorted points
            tsim = np.union1d(tsim, switchpts)
        else:
            self._simalgvars = self._algvars

        # Check if initial conditions were provided, otherwise obtain
        # them from the current variable values
        if initcon is not None:
            if len(initcon) > len(self._diffvars):
                raise ValueError(
                    "Too many initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    % len(self._diffvars))
            if len(initcon) < len(self._diffvars):
                raise ValueError(
                    "Too few initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    % len(self._diffvars))
        else:
            initcon = []
            for v in self._diffvars:
                for idx, i in enumerate(v._args):
                    if type(i) is IndexTemplate:
                        break
                initpoint = self._contset.first()
                vidx = tuple(v._args[0:idx]) + (initpoint,) + \
                       tuple(v._args[idx + 1:])
                # This line will raise an error if no value was set
                initcon.append(value(v._base[vidx]))

        # Call the integrator
        if self._intpackage is 'scipy':
            if not scipy_available:
                raise ValueError("The scipy module is not available. "
                                  "Cannot simulate the model.")
            tsim, profile = self._simulate_with_scipy(initcon, tsim, switchpts,
                                                      varying_inputs,
                                                      integrator,
                                                      integrator_options)
        else:

            if len(switchpts) != 0:
                tsim, profile = \
                    self._simulate_with_casadi_with_inputs(initcon, tsim,
                                                           varying_inputs,
                                                           integrator,
                                                           integrator_options)
            else:
                tsim, profile = \
                    self._simulate_with_casadi_no_inputs(initcon, tsim,
                                                         integrator,
                                                         integrator_options)

        self._tsim = tsim
        self._simsolution = profile
            
        return [tsim, profile]

    def _simulate_with_scipy(self, initcon, tsim, switchpts,
                             varying_inputs, integrator,
                             integrator_options):

        scipyint = \
            scipy.ode(self._rhsfun).set_integrator(integrator,
                                                   **integrator_options)
        scipyint.set_initial_value(initcon, tsim[0])

        profile = np.array(initcon)
        i = 1
        while scipyint.successful() and scipyint.t < tsim[-1]:

            # check if tsim[i-1] is a switching time and update value
            if tsim[i - 1] in switchpts:
                for v in self._siminputvars.keys():
                    if tsim[i - 1] in varying_inputs[v]:
                        p = self._templatemap[self._siminputvars[v]]
                        p.set_value(varying_inputs[v][tsim[i - 1]])

            profilestep = scipyint.integrate(tsim[i])
            profile = np.vstack([profile, profilestep])
            i += 1

        if not scipyint.successful():
            raise DAE_Error("The Scipy integrator %s did not terminate "
                            "successfully." % integrator)
        return [tsim, profile]

    def _simulate_with_casadi_no_inputs(self, initcon, tsim, integrator,
                                        integrator_options):
        # Old way (10 times faster, but can't incorporate time
        # varying parameters/controls)
        xalltemp = [self._templatemap[i] for i in self._diffvars]
        xall = casadi.vertcat(*xalltemp)

        odealltemp = [convert_pyomo2casadi(self._rhsdict[i])
                      for i in self._derivlist]
        odeall = casadi.vertcat(*odealltemp)
        dae = {'x': xall, 'ode': odeall}

        if len(self._algvars) != 0:
            zalltemp = [self._templatemap[i] for i in self._simalgvars]
            zall = casadi.vertcat(*zalltemp)

            algalltemp = [convert_pyomo2casadi(i) for i in self._alglist]
            algall = casadi.vertcat(*algalltemp)
            dae['z'] = zall
            dae['alg'] = algall

        integrator_options['grid'] = tsim
        integrator_options['output_t0'] = True
        F = casadi.integrator('F', integrator, dae, integrator_options)
        sol = F(x0=initcon)
        profile = sol['xf'].full().T

        if len(self._algvars) != 0:
            algprofile = sol['zf'].full().T
            profile = np.concatenate((profile, algprofile), axis=1)

        return [tsim, profile]

    def _simulate_with_casadi_with_inputs(self, initcon, tsim, varying_inputs,
                                          integrator, integrator_options):

        xalltemp = [self._templatemap[i] for i in self._diffvars]
        xall = casadi.vertcat(*xalltemp)

        time = casadi.SX.sym('time')

        odealltemp = [time * convert_pyomo2casadi(self._rhsdict[i])
                      for i in self._derivlist]
        odeall = casadi.vertcat(*odealltemp)

        # Time-varying inputs
        ptemp = [self._templatemap[i]
                 for i in self._siminputvars.values()]
        pall = casadi.vertcat(time, *ptemp)

        dae = {'x': xall, 'p': pall, 'ode': odeall}

        if len(self._algvars) != 0:
            zalltemp = [self._templatemap[i] for i in self._simalgvars]
            zall = casadi.vertcat(*zalltemp)
            # Need to do anything special with time scaling??
            algalltemp = [convert_pyomo2casadi(i) for i in self._alglist]
            algall = casadi.vertcat(*algalltemp)
            dae['z'] = zall
            dae['alg'] = algall

        integrator_options['tf'] = 1.0
        F = casadi.integrator('F', integrator, dae, integrator_options)
        N = len(tsim)

        # This approach removes the time scaling from tsim so must
        # create an array with the time step between consecutive
        # time points
        tsimtemp = np.hstack([0, tsim[1:] - tsim[0:-1]])
        tsimtemp.shape = (1, len(tsimtemp))

        palltemp = [casadi.DM(tsimtemp)]

        # Need a similar np array for each time-varying input
        for p in self._siminputvars.keys():
            profile = varying_inputs[p]
            tswitch = list(profile.keys())
            tswitch.sort()
            tidx = [tsim.searchsorted(i) for i in tswitch] + \
                   [len(tsim) - 1]
            ptemp = [profile[0]] + \
                    [casadi.repmat(profile[tswitch[i]], 1,
                                   tidx[i + 1] - tidx[i])
                     for i in range(len(tswitch))]
            temp = casadi.horzcat(*ptemp)
            palltemp.append(temp)

        I = F.mapaccum('simulator', N)
        sol = I(x0=initcon, p=casadi.vertcat(*palltemp))
        profile = sol['xf'].full().T

        if len(self._algvars) != 0:
            algprofile = sol['zf'].full().T
            profile = np.concatenate((profile, algprofile), axis=1)

        return [tsim, profile]

    def initialize_model(self):
        """
        This function will initialize the model using the profile obtained
        from simulating the dynamic model.
        """
        if self._tsim is None:
            raise DAE_Error(
                "Tried to initialize the model without simulating it first")

        tvals = list(self._contset)
 
        # Build list of state and algebraic variables
        # that can be initialized
        initvars = self._diffvars + self._simalgvars
               
        for idx, v in enumerate(initvars):
            for idx2, i in enumerate(v._args):
                    if type(i) is IndexTemplate:
                        break
            valinit = np.interp(tvals, self._tsim,
                                self._simsolution[:, idx])
            for i, t in enumerate(tvals):
                vidx = tuple(v._args[0:idx2]) + (t,) + \
                       tuple(v._args[idx2 + 1:])
                v._base[vidx] = valinit[i]
