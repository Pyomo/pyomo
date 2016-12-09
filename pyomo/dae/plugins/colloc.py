#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
from six import itervalues
from six.moves import xrange

from pyomo.core.base.plugin import alias
from pyomo.core.base import Transformation
from pyomo.core import *
from pyomo.dae import *
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import update_contset_indexed_component
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information

# If the user has numpy then the collocation points and the a matrix for
# the Runge-Kutta basis formulation will be calculated as needed.
# If the user does not have numpy then these values will be read from a
# stored dictionary for up to 10 collocation points.
try:
    import numpy
    numpy_available=True
except ImportError:
    numpy_available=False
    from pyomo.dae.utilities import *

logger = logging.getLogger('pyomo.core')

def _lagrange_radau_transform(v,s):
    ncp = s.get_discretization_info()['ncp']
    adot = s.get_discretization_info()['adot']
    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0: # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx+j])*adot[j][idx-lowidx]*(1.0/(tmp[lowidx+ncp]-tmp[lowidx])) for j in range(ncp+1))
    return _fun

def _lagrange_radau_transform_order2(v,s):
    ncp = s.get_discretization_info()['ncp']
    adotdot = s.get_discretization_info()['adotdot']
    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0: # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx+j])*adotdot[j][idx-lowidx]*(1.0/(tmp[lowidx+ncp]-tmp[lowidx])**2) for j in range(ncp+1))
    return _fun

def _lagrange_legendre_transform(v,s):
    ncp = s.get_discretization_info()['ncp']
    adot = s.get_discretization_info()['adot']
    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0: # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        elif i in s.get_finite_elements(): # Don't apply at finite element points
                                           # continuity equations added later
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx+j])*adot[j][idx-lowidx]*(1.0/(tmp[lowidx+ncp+1]-tmp[lowidx])) for j in range(ncp+1))
    return _fun

def _lagrange_legendre_transform_order2(v,s):
    ncp = s.get_discretization_info()['ncp']
    adotdot = s.get_discretization_info()['adotdot']
    def _fun(i):
        tmp = sorted(s)
        idx = tmp.index(i)
        if idx == 0: # Don't apply this equation at initial point
            raise IndexError("list index out of range")
        elif i in s.get_finite_elements(): # Don't apply at finite element points
                                           # continuity equations added later
            raise IndexError("list index out of range")
        low = s.get_lower_element_boundary(i)
        lowidx = tmp.index(low)
        return sum(v(tmp[lowidx+j])*adotdot[j][idx-lowidx]*(1.0/(tmp[lowidx+ncp+1]-tmp[lowidx])**2) for j in range(ncp+1))
    return _fun

def _hermite_cubic_transform(v,s):
    pass

def factorial(n):
    if n < 0 or not isinstance(n,int):
        raise ValueError("Can only take the factorial of a non-negative integer")
    elif n > 0:
        return n*factorial(n-1)
    else:
        return 1

def conv(a,b):
    if len(a)==0 or len(b)==0:
        raise ValueError("Cannot convolve an empty list")

    ans = []
    m = len(a)
    n = len(b)

    for k in range(m+n-1):
        val = 0
        j = max(0,k-n)
        stop = min(k,m)
        while j<=stop:
             if j<m and (k-j)<n:
                val += a[j]*b[k-j]
             j += 1
        ans.insert(k,val)

    return ans

def calc_cp(alpha,beta,k):
    gamma = []
    for i in range(k+1):
        num = factorial(alpha+k)*factorial(alpha+beta+k+i)
        denom = factorial(alpha+i)*factorial(k-i)*factorial(i)
        gamma.insert(i,num/denom)

    poly = []
    for i in range(k+1):
        if i == 0:
            poly.insert(i,gamma[i])
        else:
            prod = [1]
            j=1
            while j<=i:
                prod=conv(prod,[1,-1])
                j=j+1
            while len(poly)<len(prod):
                poly.insert(0,0)
            prod = [gamma[i]*t for t in prod]
            poly = [sum(pair) for pair in zip(poly,prod)]

    cp = numpy.roots(poly)
    return cp

def calc_omega(cp):
    cp.insert
    a=[]
    for i in range(len(cp)):
        ptmp = []
        tmp = 0
        for j in range(len(cp)):
            if j != i:
                row = []
                row.insert(0,1/(cp[i]-cp[j]))
                row.insert(1,-cp[j]/(cp[i]-cp[j]))
                ptmp.insert(tmp,row)
                tmp += 1
        p=[1]
        for j in range(len(cp)-1):
            p = conv(p,ptmp[j])
        pint = numpy.polyint(p)
        arow = []
        for j in range(len(cp)):
            arow.append(numpy.polyval(pint,cp[j]))
        a.append(arow)
    return a

def calc_adot(cp,order=1):
    a=[]
    for i in range(len(cp)):
        ptmp = []
        tmp = 0
        for j in range(len(cp)):
            if j != i:
                row = []
                row.insert(0,1/(cp[i]-cp[j]))
                row.insert(1,-cp[j]/(cp[i]-cp[j]))
                ptmp.insert(tmp,row)
                tmp += 1
        p=[1]
        for j in range(len(cp)-1):
            p = conv(p,ptmp[j])
        pder = numpy.polyder(p,order)
        arow = []
        for j in range(len(cp)):
            arow.append(numpy.polyval(pder,cp[j]))
        a.append(arow)
    return a

def calc_afinal(cp):
    afinal=[]
    for i in range(len(cp)):
        ptmp = []
        tmp = 0
        for j in range(len(cp)):
            if j != i:
                row = []
                row.insert(0,1/(cp[i]-cp[j]))
                row.insert(1,-cp[j]/(cp[i]-cp[j]))
                ptmp.insert(tmp,row)
                tmp += 1
        p=[1]
        for j in range(len(cp)-1):
            p = conv(p,ptmp[j])
        afinal.append(numpy.polyval(p,1.0))
    return afinal

class Collocation_Discretization_Transformation(Transformation):

    alias('dae.collocation', doc="Discretizes a DAE model using "\
          "orthogonal collocation over finite elements transforming "\
          "the model into an NLP.")

    def __init__(self):
        super(Collocation_Discretization_Transformation, self).__init__()
        self._ncp = {}
        self._nfe = {}
        self._adot = {}
        self._adotdot = {}
        self._afinal = {}
        self._tau = {}
        self._reduced_cp = {}
        self.all_schemes = {
            'LAGRANGE-RADAU' : (_lagrange_radau_transform, _lagrange_radau_transform_order2),
            'LAGRANGE-LEGENDRE' : (_lagrange_legendre_transform,_lagrange_legendre_transform_order2),
            'HERMITE-CUBIC' : _hermite_cubic_transform,
            }

    def _setup(self, instance):
        instance = instance.clone()
        instance.construct()
        return instance

    def _get_radau_constants(self,currentds):
        """
        This function sets the radau collocation points and a values depending
        on how many collocation points have been specified and whether or not
        the user has numpy
        """
        if not numpy_available:
            if self._ncp[currentds] > 10:
                raise ValueError("Numpy was not found so the maximum number of "\
                    "collocation points is 10")
            self._tau[currentds] = radau_tau_dict[self._ncp[currentds]]
            self._adot[currentds] = radau_adot_dict[self._ncp[currentds]]
            self._adotdot[currentds] = radau_adotdot_dict[self._ncp[currentds]]
            self._afinal[currentds] = None
        else:
            alpha = 1
            beta = 0
            k = self._ncp[currentds]-1
            cp = sorted(list(calc_cp(alpha,beta,k)))
            cp.insert(0,0.0)
            cp.append(1.0)
            adot = calc_adot(cp,1)
            adotdot = calc_adot(cp,2)

            self._tau[currentds] = cp
            self._adot[currentds] = adot
            self._adotdot[currentds] = adotdot
            self._afinal[currentds] = None

    def _get_legendre_constants(self,currentds):
        """
        This function sets the legendre collocation points and a values depending
        on how many collocation points have been specified and whether or not
        the user has numpy
        """
        if not numpy_available:
            if self._ncp[currentds] > 10:
                raise ValueError("Numpy was not found so the maximum number of "\
                    "collocation points is 10")
            self._tau[currentds] = legendre_tau_dict[self._ncp[currentds]]
            self._adot[currentds] = legendre_adot_dict[self._ncp[currentds]]
            self._adotdot[currentds] = legendre_adotdot_dict[self._ncp[currentds]]
            self._afinal[currentds] = legendre_afinal_dict[self._ncp[currentds]]
        else:
            alpha = 0
            beta = 0
            k = self._ncp[currentds]
            cp = sorted(list(calc_cp(alpha,beta,k)))
            cp.insert(0,0.0)
            adot = calc_adot(cp,1)
            adotdot = calc_adot(cp,2)
            afinal = calc_afinal(cp)

            self._tau[currentds] = cp
            self._adot[currentds] = adot
            self._adotdot[currentds] = adotdot
            self._afinal[currentds] = afinal

    def _get_hermite_constants(self,currentds):
        # TODO: finish this
        raise DAE_Error("Not Implemented")

    def _apply_to(self, instance, **kwds):
        """
        Applies specified collocation transformation to a modeling instance

        Keyword Arguments:
        nfe           The desired number of finite element points to be
                      included in the discretization.
        ncp           The desired number of collocation points over each
                      finite element.
        wrt           Indicates which ContinuousSet the transformation
                      should be applied to. If this keyword argument is not
                      specified then the same scheme will be applied to all
                      ContinuousSets.
        scheme        Indicates which finite difference method to apply.
                      Options are LAGRANGE-RADAU, LAGRANGE-LEGENDRE, or
                      HERMITE-CUBIC. The default scheme is Lagrange polynomials
                      with Radau roots.
        """

        options = kwds.pop('options', {})

        tmpnfe = kwds.pop('nfe',10)
        tmpncp = kwds.pop('ncp',3)
        tmpds = kwds.pop('wrt',None)
        tmpscheme = kwds.pop('scheme','LAGRANGE-RADAU')
        self._scheme_name = tmpscheme.upper()

        if tmpds is not None:
            if tmpds.type() is not ContinuousSet:
                raise TypeError("The component specified using the 'wrt' keyword "\
                     "must be a differential set")
            elif 'scheme' in tmpds.get_discretization_info():
                raise ValueError("The discretization scheme '%s' has already been applied "\
                     "to the ContinuousSet '%s'"%(tmpds.get_discretization_info()['scheme'],tmpds.name))

        if tmpnfe <=0:
            raise ValueError("The number of finite elements must be at least 1")
        if tmpncp <= 0:
            raise ValueError("The number of collocation points must be at least 1")

        if None in self._nfe:
            raise ValueError("A general discretization scheme has already been applied to "\
                    "to every differential set in the model. If you would like to specify a "\
                    "specific discretization scheme for one of the differential sets you must discretize "\
                    "each differential set individually. If you would like to apply a different "\
                    "discretization scheme to all differential sets you must declare a new Collocation"\
                    "_Discretization object")

        if len(self._nfe) == 0 and tmpds is None:
            # Same discretization on all differentialsets
            self._nfe[None] = tmpnfe
            self._ncp[None] = tmpncp
            currentds = None
        else :
            self._nfe[tmpds.local_name]=tmpnfe
            self._ncp[tmpds.local_name]=tmpncp
            currentds = tmpds.name

        self._scheme = self.all_schemes.get(self._scheme_name,None)
        if self._scheme is None:
            raise ValueError("Unknown collocation scheme '%s' specified using the "\
                     "'scheme' keyword. Valid schemes are 'LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE'"\
                     ", and 'HERMITE-CUBIC'" %(tmpscheme))

        if self._scheme_name == 'LAGRANGE-RADAU':
            self._get_radau_constants(currentds)
        elif self._scheme_name == 'LAGRANGE-LEGENDRE':
            self._get_legendre_constants(currentds)

        for block in instance.block_data_objects(active=True):
            self._transformBlock(block,currentds)

        return instance

    def _transformBlock(self, block, currentds):

        self._fe = {}
        for ds in itervalues(block.component_map(ContinuousSet)):
            if currentds is None or currentds == ds.name:
                generate_finite_elements(ds,self._nfe[currentds])
                if not ds.get_changed():
                    if len(ds)-1 > self._nfe[currentds]:
                        print("***WARNING: More finite elements were found in differentialset "\
                            "'%s' than the number of finite elements specified in apply. "\
                              "The larger number of finite elements will be used." % (ds.name,))

                self._nfe[ds.name]=len(ds)-1
                self._fe[ds.name]=sorted(ds)
                generate_colloc_points(ds,self._tau[currentds])
                # Adding discretization information to the differentialset object itself
                # so that it can be accessed outside of the discretization object
                disc_info = ds.get_discretization_info()
                disc_info['nfe']=self._nfe[ds.name]
                disc_info['ncp']=self._ncp[currentds]
                disc_info['tau_points']=self._tau[currentds]
                disc_info['adot'] = self._adot[currentds]
                disc_info['adotdot'] = self._adotdot[currentds]
                disc_info['afinal'] = self._afinal[currentds]
                disc_info['scheme'] = self._scheme_name

        for c in itervalues(block.component_map()):
            update_contset_indexed_component(c)

        for d in itervalues(block.component_map(DerivativeVar)):
            dsets = d.get_continuousset_list()
            for i in set(dsets):
                if currentds is None or i.name == currentds:
                    oldexpr = d.get_derivative_expression()
                    loc = d.get_state_var()._contset[i]
                    count = dsets.count(i)
                    if count >= 3:
                        raise DAE_Error(
                            "Error discretizing '%s' with respect to '%s'. Current implementation "\
                            "only allows for taking the first or second derivative with respect to "\
                            "a particular ContinuousSet" %(d.name,i.name))
                    scheme = self._scheme[count-1]
                    # print("%s %s" % (i.name, scheme.__name__))
                    newexpr = create_partial_expression(scheme,oldexpr,i,loc)
                    d.set_derivative_expression(newexpr)
                    if self._scheme_name == 'LAGRANGE-LEGENDRE':
                        add_continuity_equations(block,d,i,loc)

            # Reclassify DerivativeVar if all indexing ContinuousSets have been discretized
            if d.is_fully_discretized():
                add_discretization_equations(block,d)
                block.reclassify_component_type(d,Var)

        # Reclassify Integrals if all ContinuousSets have been discretized
        if block_fully_discretized(block):

            if block.contains_component(Integral):
                for i in itervalues(block.component_map(Integral)):
                    i.reconstruct()
                    block.reclassify_component_type(i,Expression)
                # If a model contains integrals they are most likely to appear in the objective
                # function which will need to be reconstructed after the model is discretized.
                for k in itervalues(block.component_map(Objective)):
                    k.reconstruct()

    def _get_idx(self,l,t,n,i,k):
        """
        This function returns the appropriate index for the differential
        and the derivative variables. It's needed because the collocation
        constraints are indexed by finite element and collocation point
        however a differentialset contains a list of all the discretization
        points and is not separated into finite elements and collocation
        points.
        """

        tmp = t.index(t._fe[i])
        tik = t[tmp+k]
        if n is None:
            return tik
        else:
            tmpn=n
            if not isinstance(n,tuple):
                tmpn = (n,)
            return tmpn[0:l]+(tik,)+tmpn[l:]

    def reduce_collocation_points(self, instance, var=None, ncp=None, contset=None):
        """
        This method will add additional constraints to a model if some
        of the Variables are specified as having less collocation points
        than the default
        """
        if contset is None:
            raise TypeError("A continuous set must be specified using the keyword 'contset'")
        if contset.type() is not ContinuousSet:
            raise TypeError("The component specified using the 'contset' keyword "\
                "must be a differential set")
        ds = instance.find_component(contset.name)
        if ds is None:
            raise ValueError("ContinuousSet '%s' is not a valid component of the discretized "\
                "model instance" %(contset.name))

        if len(self._ncp) == 0:
            raise RuntimeError("This method should only be called after using the apply() method "\
                "to discretize the model")
        elif None in self._ncp:
            tot_ncp = self._ncp[None]
        elif ds.name in self._ncp:
            tot_ncp = self._ncp[ds.name]
        else:
            raise ValueError("ContinuousSet '%s' has not been discretized yet, please call "\
                "the apply() method with this ContinuousSet to discretize it before calling "\
                "this method" %(ds.name))

        if var is None:
            raise TypeError("A variable must be specified")
        if var.type() is not Var:
            raise TypeError("The component specified using the 'var' keyword "\
                "must be a variable")
        tmpvar = instance.find_component(var.name)
        if tmpvar is None:
            raise ValueError("Variable '%s' is not a valid component of the discretized "\
                "model instance" %(var.name))

        var = tmpvar

        if ncp is None:
            raise TypeError("The number of collocation points must be specified")
        if ncp <= 0:
            raise ValueError("The number of collocation points must be at least 1")
        if ncp > tot_ncp:
            raise ValueError("The number of collocation points used to interpolate "\
                "an individual variable must be less than the number used to discretize "\
                "the original model")
        if ncp == tot_ncp:
            # Nothing to be done
            return instance

        # Check to see if the continuousset is an indexing set of the variable
        if var.dim() == 1:
            if ds not in var._index:
                raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'"\
                    % (ds.name, var.name))
        elif ds not in var._index_set:
            raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'"\
                             % (ds.name, var.name))

        if var.name in self._reduced_cp:
            temp = self._reduced_cp[var.name]
            if ds.name in temp:
                raise RuntimeError("Variable '%s' has already been constrained to a reduced "\
                    "number of collocation points over ContinuousSet '%s'.")
            else:
                temp[ds.local_name]=ncp
        else:
            self._reduced_cp[var.local_name] = {ds.local_name: ncp}

        list_name = var.local_name+"_interpolation_constraints"

        instance.add_component(list_name,ConstraintList())
        conlist = instance.find_component(list_name)

        t = sorted(ds)
        fe = ds._fe
        info = get_index_information(var,ds)
        tmpidx = info['non_ds']
        idx = info['index function']

        # Iterate over non_ds indices
        for n in tmpidx:
            # Iterate over finite elements
            for i in xrange(0,len(fe)-1):
                # Iterate over collocation points
                for k in xrange(1,tot_ncp-ncp+1):
                    if ncp == 1:
                        # Constant over each finite element
                        conlist.add(var[idx(n,i,k)]==var[idx(n,i,tot_ncp)])
                    else:
                        tmp = t.index(fe[i])
                        tmp2 = t.index(fe[i+1])
                        ti = t[tmp+k]
                        tfit = t[tmp2-ncp+1:tmp2+1]
                        coeff = self._interpolation_coeffs(ti,tfit)
                        conlist.add(var[idx(n,i,k)]== sum(var[idx(n,i,j)]*coeff.next() for j in xrange(tot_ncp-ncp+1,tot_ncp+1)))

        return instance

    def _interpolation_coeffs(self,ti,tfit):

        for i in tfit:
            l=1
            for j in tfit:
                if i != j:
                    l = l*(ti-j)/(i-j)
            yield l

