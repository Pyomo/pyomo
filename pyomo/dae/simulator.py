#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('Simulator', )

import matplotlib.pyplot as plt
import numpy as np

from pyomo.environ import Constraint, Param, value

from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error

from pyomo.core.base.component import register_component
from pyomo.core.base import expr as EXPR
from pyomo.core.base import expr_common as common
from pyomo.core.base.template_expr import (
    IndexTemplate,
    substitute_template_expression,
    substitute_template_with_param,
    substitute_template_with_index,
)

def _check_getitemexpression(expr,i):
    """
    Accepts an equality expression and an index value. Checks the
    GetItemExpression at expr._args[i] to see if it is a
    DerivativeVar. If so, return the GetItemExpression for the
    DerivativeVar and the RHS. If not, return None.
    """
    if type(expr._args[i]._base) is DerivativeVar:
        return [expr._args[i],expr._args[i-1]]
    else:
        return None

def _check_productexpression(expr,i):
    """
    Accepts an equality expression and an index value. Checks the
    ProductExpression at expr._args[i] to see if it contains a
    DerivativeVar. If so, return the GetItemExpression for the
    DerivativeVar and the RHS. If not, return None.
    """
    if common.mode is common.Mode.coopr3_trees:
        num = expr._args[i]._numerator
        denom = expr._args[i]._denominator
        coef = expr._args[i]._coef

        dv = None
        # Check if there is a DerivativeVar in the numerator
        for idx, obj in enumerate(num):
            if type(obj) is EXPR._GetItemExpression and \
               type(obj._base) is DerivativeVar:
                dv = obj
                num = num[0:idx]+num[idx+1:]
                break
        if dv is not None:
            RHS = expr._args[i-1]/coef
            for obj in num:
                RHS = RHS*(1/obj)
            for obj in denom:
                RHS = RHS*obj
            return [dv, RHS]
        else:
            # Check if there is a DerivativeVar in the denominator
            for idx, obj in enumerate(denom):
                if type(obj) is EXPR._GetItemExpression and \
                   type(obj._base) is DerivativeVar:
                    dv = obj
                    denom = denom[0:idx]+denom[idx+1:]
                    break
            if dv is not None:
                tempnum = coef
                for obj in num:
                    tempnum *= obj

                tempdenom = expr._args[i-1]
                for obj in denom:
                    tempdenom *= obj

                RHS = tempnum/tempdenom
                return [dv, RHS]                
    else:
        raise TypeError(
            "Simulator is unable to handle pyomo4 expression trees.")
    return None

def _check_sumexpression(expr,i):
    """
    Accepts an equality expression and an index value. Checks the
    SumExpression at expr._args[i] to see if it contains a
    DerivativeVar. If so, return the GetItemExpression for the
    DerivativeVar and the RHS. If not, return None.
    """
    sumexp = expr._args[i]
    items = sumexp._args
    coefs = sumexp._coef
    dv = None
    dvcoef = 1

    for idx,item in enumerate(items):
        if type(item) is EXPR._GetItemExpression and \
           type(item._base) is DerivativeVar:
            dv = item
            dvcoef = coefs[idx]
            items = items[0:idx]+items[idx+1:]
            coefs = coefs[0:idx]+coefs[idx+1:]
            break

    if dv is not None:
        RHS = expr._args[i-1]
        for idx,item in enumerate(items):
            RHS -= coefs[idx]*item
        RHS = RHS/dvcoef
        return [dv, RHS]

    return None

class Simulator:
    """
    Simulator objects allow a user to simulate a dynamic model
    formulated using pyomo.dae.
    """

    def __init__(self, m, **kwds):
        
        self._intpackage = kwds.pop('package','scipy')
        if self._intpackage != 'scipy':
            raise DAE_Error(
                "The integrator package %s was specified using the "
                "'package' keyword argument. SciPy is the only "
                "package currently supported by the "
                "Simulator."%(self._intpackage))

        temp = m.component_map(ContinuousSet)
        if len(temp) != 1:
            raise DAE_Error(
                "Currently the scipy integrator may only be applied to "
                "Pyomo models with a single ContinuousSet")

        # Get the ContinuousSet in the model
        contset = temp.values()[0]

        # Create a index template for the continuous set
        cstemplate = IndexTemplate(contset)

        derivs = m.component_map(DerivativeVar)
        if len(derivs) == 0:
            raise DAE_Error("Cannot simulate a model with no derivatives")

        templatemap = {} # Map for template substituter 
        rhsdict = {} # Map of derivative to its RHS templated expr
        derivlist = []  # Ordered list of derivatives

        # Loop over constraints to find differential equations with separable
        # RHS. Must find a RHS for every derivative var otherwise ERROR. Build
        # dictionary of DerivativeVar:RHS equation. In simple ODE case can
        # populate the vmap directly from the derivative var since it
        # shouldn't include any other algebraic variables
        for con in m.component_objects(Constraint):
            
            # Skip the discretization equations if model is discretized
            if '_disc_eq' in con.name:
                continue
            
            # Check dimension of the Constraint. Current
            # implementation only works for single dimensional
            # Constraints. 
            if con.dim() == 0:
                continue
            elif con.dim() > 1:
                print(
                    "WARNING: Any differential equations indexed by "
                    "multiple sets will not be simulated.")
                continue

            # Check if the continuous set is the indexing set
            if not con._index is contset :
                continue
            
            # Get the rule used to construct the constraint
            conrule = con.rule

            # Call the rule with the IndexTemplate to create a
            # templated expression
            tempexp = conrule(m,cstemplate)

            # Check to make sure it's an _EqualityExpression
            if not type(tempexp) is EXPR._EqualityExpression:
                continue
            
            # Check to make sure it's a differential equation with
            # separable RHS
            args = None
            # Case 1: m.dxdt[t] = RHS 
            if type(tempexp._args[0]) is EXPR._GetItemExpression:
                args = _check_getitemexpression(tempexp,0)
            
            # Case 2: RHS = m.dxdt[t]
            if args is None:
                if type(tempexp._args[1]) is EXPR._GetItemExpression:
                    args = _check_getitemexpression(tempexp,1)

            # Case 3: m.p*m.dxdt[t] = RHS
            if args is None:
                if type(tempexp._args[0]) is EXPR._ProductExpression:
                    args = _check_productexpresstion(tempexp,0)

            # Case 4: RHS =  m.p*m.dxdt[t]
            if args is None:
                if type(tempexp._args[1]) is EXPR._ProductExpression:
                    args = _check_productexpresstion(tempexp,1)

            # Case 5: m.dxdt[t] + CONSTANT = RHS or CONSTANT + m.dxdt[t] = RHS
            if args is None:
                if type(tempexp._args[0]) is EXPR._SumExpression:
                    args = _check_sumexpresstion(tempexp,0)

            # Case 6: RHS = m.dxdt[t] + CONSTANT
            if args is None:
                if type(tempexp._args[1]) is EXPR._SumExpression:
                    args = _check_sumexpresstion(tempexp,1)

            
            # Case 7: RHS = m.p*m.dxdt[t] + CONSTANT
            # This case will be caught by Case 6 if p is immutable. If
            # p is mutable then this case will not be detected as a
            # separable differential equation

            # At this point if args is not None then args[0] contains
            # the _GetItemExpression for the DerivativeVar and args[1]
            # contains the RHS expression
            if args is None:
                # Constraint is not a separable differential equation
                continue
                
            
            # Add the differential equation to rhsdict and derivlist
            dv = args[0]
            RHS = args[1]
            _name = dv._base.name
            if _name in rhsdict:
                raise DAE_Error(
                    "Found multiple RHS expressions for the "
                    "DerivativeVar %s" %(_name))
            
            derivlist.append(_name)
            rhsdict[_name] = substitute_template_expression(
                RHS, substitute_template_with_param, templatemap)
        
        # Check to see if we found a RHS for every DerivativeVar in
        # the model
        allderivs = derivs.keys()
        if set(allderivs) != set(derivlist):
            missing = list(set(allderivs)-set(derivlist))
            print("WARNING: Could not find a RHS expression for the "
            "following DerivativeVar components "+str(missing))

        # Create ordered list of differential variables corresponding
        # to the list of derivatives.
        diffvars=[]
        diffvarids=[]
        for deriv in derivlist:
            deriv = m.component(deriv)
            diffvars.append(deriv.get_state_var().name)
            diffvarids.append(id(deriv.get_state_var()))

        # Make sure there are no DerivativeVars or algebraic variables
        # in the RHS expressions. The template map should only contain
        # differential variables.
        for item in templatemap.values():
            if item.name in allderivs:
                raise DAE_Error(
                    "Cannot simulate a differential equation with "
                    "multiple DerivativeVars")
            if item.name not in diffvars:
                # This only catches algebraic variables indexed by
                # time. TODO: how to catch variables not indexed by
                # time or warn the user that values must be set for
                # these variables.
                raise DAE_Error(
                    "Cannot simulate a differential equation with "
                    "algebraic variables")

        # Function sent to scipy integrator
        def _rhsfun(x,t):
            residual = []
            cstemplate.set_value(t)
            for idx,v in enumerate(diffvarids):
                if v in templatemap:
                    templatemap[v].set_value(x[idx])

            for d in derivlist:
                residual.append(rhsdict[d]())

            return residual
            
        self._contset = contset
        self._cstemplate = cstemplate
        self._diffvars = diffvars
        self._diffvarids = diffvarids
        self._derivlist = derivlist
        self._templatemap = templatemap
        self._rhsdict = rhsdict
        self._rhsfun = _rhsfun   
        self._model = m
        self._tsim = None
        self._simsolution = None

    def get_variable_order(self):
        """
        This function returns the ordered list of differential variable
        names. The order corresponds to the order they are being sent
        to the integrator function. Knowing the order allows users to
        provide initial conditions for the differential equations
        using a list.
        """        
        return self._diffvars
        
    def simulate(self,**kwds):
        """
        Simulate the model. The user may specify initial conditions using
        a list. If no list is provided then the simulator will take
        the current value of the differential variable at the lower
        bound of the ContinuousSet. The user may also specify the time
        step for the profiles returned by the integrator and other
        simulation options.
        """

        tstep = kwds.pop('step', None)
        if tstep is not None and \
           tstep > (self._contset.last()-self._contset.first()):
            raise ValueError(
                "The step size %6.2f is larger than the span of the "
                "ContinuousSet %s" %(tstep,self._contset.name()))
            
        numpoints = kwds.pop('numpoints',None)
        
        if tstep is not None and numpoints is not None:
            raise ValueError(
                "Cannot specify both the step size and the number of "
                "points for the simulator")
        if tstep is None and numpoints is None:
            # Use 100 points by default
            numpoints = 100

        tsim = []
        if tstep is None:
            tsim = np.linspace(
                self._contset.first(),self._contset.last(),numpoints)
        else:
            tsim = np.arrange(
                self._contset.first(),self._contset.last(),tstep)

        initcon = kwds.pop('initcon',None)
        if initcon is not None:
            if len(initcon)>len(self._diffvars):
                raise ValueError(
                    "Too many initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    %(len(self._diffvars)))
            if len(initcon)<len(self._diffvars):
                raise ValueError(
                    "Too few initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    %(len(self._diffvars)))
        else:
            initcon = []
            for nme in self._diffvars:
                v = self._model.component(nme)
                # This line will raise an error if no value was set
                initcon.append(value(v[self._contset.first()]))

        try:
            from scipy.integrate import odeint
        except ImportError:
            raise ImportError("Tried to import SciPy but failed")
        
        # TODO: Figure out how to get information back from integrator and
        # verify that it terminated successfully
        sol = odeint(self._rhsfun,initcon,tsim)

        self._tsim = tsim
        self._simsolution = sol
            
        return [tsim,sol]

    def initialize_model(self):
        """
        This function will initialize the model using the profile
        obtained from the simulating the ODE system.
        """
        if self._tsim is None:
            raise DAE_Error(
                "Tried to initialize the model without simulating it "
                "first")

        tvals = list(self._contset)
        
        for idx,nme in enumerate(self._diffvars):
            v = self._model.component(nme)
            valinit = np.interp(tvals, self._tsim,
                                self._simsolution[:,idx])
            for i,t in enumerate(tvals):
                v[t] = valinit[i]

register_component(Simulator, "Used to simulate a system of ODEs")
