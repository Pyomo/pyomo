#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import logging
import sys

from pyomo.misc.plugin import alias
from pyomo.core.base import Transformation
from pyomo.core import *
from pyomo.core.base import Block
from pyomo.dae import *
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import update_diffset_indexed_component
from pyomo.dae.misc import add_equality_constraints
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

class Collocation_Discretization_Transformation(Transformation):

    alias('dae.collocation_discretization', doc="TODO")

    def __init__(self):
        super(Collocation_Discretization_Transformation, self).__init__()

    def _setup(self, instance):
        instance = instance.clone()
        instance.concrete_mode()
        # Note: all_blocks is a generator and since we are changing the
        # type of Differential components to Block after the discretization,
        # we have to explicitly form the list of all Blocks before iterating
        # over them. Otherwise, the differential components will be included
        # in the iteration and _transformBlock will be called unintentionally 
        # for each differential component after their type has been changed.
        self._blocks = list(instance.all_blocks())
        self._ncp = {}
        self._nfe = {}
        self._a = {}
        self._tau = {}
        self._reduced_cp = {}

        # This is a single attribute of the discretization object because I don't think
        # you'd ever want to mix radau and legendre collocation schemes on a single
        # model. If this is a bad assumption _radau would be changed to a dictionary
        # similar to _a or _tau
        # TODO: Add option of doing Legendre collocation which will require some fancy 
        # interpolating, 'type' will then become a keyword argument for init with the 
        # possible values of 'radau' or 'legendre'.
        self._radau = True
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
            self._a[currentds] = radau_a_dict[self._ncp[currentds]]
        else:
            alpha = 1
            beta = 0
            k = self._ncp[currentds]-1
            cp = sorted(list(calc_cp(alpha,beta,k)))
            cp.append(1.0)
            a = calc_omega(cp)
            self._tau[currentds] = cp
            self._a[currentds] = a

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
            self._a[currentds] = legendre_a_dict[self._ncp[currentds]]
        else:
            alpha = 0
            beta = 0
            k = self._ncp[currentds]
            # TODO: finish this

    def apply(self, instance, **kwds):
        instance = self._setup(instance)
        tmpnfe = kwds.pop('nfe',10)
        tmpncp = kwds.pop('ncp',3)        
        tmpds = kwds.pop('diffset',None)

        if tmpds is not None and tmpds.type() is not DifferentialSet:
            raise TypeError("The component specified using the 'diffset' keyword "\
                "must be a differential set")
        if tmpnfe <=0:
            raise ValueError("The number of finite elements must be at least 1")
        if tmpncp <= 0:
            raise ValueError("The number of collocation points must be at least 1")
        
        if self._nfe.has_key(None):
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
        elif self._nfe.has_key(tmpds.name):
            raise ValueError("A discretization scheme has already been specified "\
                    "for differentialset '%s'" %s(tmpds.name))
        else :
            self._nfe[tmpds.name]=tmpnfe
            self._ncp[tmpds.name]=tmpncp
            currentds = tmpds.name
              
        if self._radau:
            self._get_radau_constants(currentds)
        else:
            self._get_legendre_constants(currentds)

        for block in self._blocks:
            self._transformBlock(block,currentds)

        # Taken from bigm
        # REQUIRED: re-call preprocess()
        instance.preprocess()
        return instance

    def _transformBlock(self, block, currentds):
        
        self._fe = {}
        for ds in block.components(DifferentialSet).itervalues():
            if currentds is None or currentds is ds.name:
                generate_finite_elements(ds,self._nfe[currentds])
                if not ds.get_changed():
                    if len(ds)-1 > self._nfe[currentds]:
                        print("***WARNING: More finite elements were found in differentialset "\
                            "'%s' than the number of finite elements specified in apply. "\
                            "The larger number of finite elements will be used." %s(ds.name))
                
                self._nfe[ds.name]=len(ds)-1
                self._fe[ds.name]=sorted(ds)
                #ds._fe=sorted(ds)
                generate_colloc_points(ds,self._tau[currentds],self._radau)
                # Adding discretization information to the differentialset object itself
                # so that it can be accessed outside of the discretization object
                disc_info = ds.get_discretization_info()
                disc_info['nfe']=self._nfe[ds.name]
                disc_info['ncp']=self._ncp[currentds]
                disc_info['tau_points']=self._tau[currentds]
                if self._radau:
                    disc_info['scheme'] = 'Radau Collocation'
                else:
                    disc_info['scheme'] = 'Legendre Collocation'
                
        
        for c in block.components().itervalues():
            if c.type() != Differential:
                update_diffset_indexed_component(c)
            else:
                update_diffset_indexed_component(c._lhs_var)

        for d in block.components(Differential).itervalues():
            if currentds is None or d.get_differentialset().name is currentds:
                add_equality_constraints(d)
                self._radau_transform(d,currentds)
                block.reclassify_component_type(d,Block)

    def _radau_transform(self,diff,currentds):
        # generate the constraints unique to this discretization scheme
        # The needed constraints are:
        # 1) Z_ik = Z_(i-1)+h_i*sum(a_kj*Zdot_ij for j in K), k = 1,...,K
        # Note: because the index specified by giving the finite element
        # and collocation point is directly related to an index in the 
        # differentialset, no additional continuity equations are required.
        ds = diff.get_differentialset()
        dv = diff.get_diffvar()
        ncp = self._ncp[currentds]
        a = self._a[currentds]
        fe = sorted(self._fe[ds.name])

        info = get_index_information(dv,ds)
        tmpidx = info['non_ds']
        idx = info['index function']

        # Iterate over non_ds indices
        for n in tmpidx:
            # Iterate over finite elements
            for i in xrange(0,len(fe)-1):
                # Iterate over collocation points
                for k in xrange(1,ncp+1):
                    # index in a starts at 0 which is why we subtract 1
                    diff._cons.add(dv[idx(n,i,k)]==dv[idx(n,i,0)] \
                                       + (fe[i+1]-fe[i])*sum(a[j-1][k-1]*diff[idx(n,i,j)] for j in xrange(1,ncp+1)))

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
          

    def reduce_collocation_points(self, instance, var=None, ncp=None, diffset=None):
        """
        This method will add additional constraints to a model if some
        of the Variables are specified as having less collocation points
        than the default
        """
        if diffset is None:
            raise TypeError("A differential set must be specified")
        if diffset.type() is not DifferentialSet:
            raise TypeError("The component specified using the 'diffset' keyword "\
                "must be a differential set")
        ds = instance.find_component(diffset.name)
        if ds is None:
            raise ValueError("DifferentialSet '%s' is not a valid component of the discretized "\
                "model instance" %(diffset.name))

        if len(self._ncp) == 0:
            raise RuntimeError("This method should only be called after using the apply() method "\
                "to discretize the model")
        elif self._ncp.has_key(None):
            tot_ncp = self._ncp[None]
        elif self._ncp.has_key(ds.name):
            tot_ncp = self._ncp[ds.name]
        else:
            raise ValueError("DifferentialSet '%s' has not been discretized yet, please call "\
                "the apply() method with this DifferentialSet to discretize it before calling "\
                "this method" %s(ds.name))

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

        # Check to see if the differentialset is an indexing set of the variable
        if var.dim() == 1:
            if ds not in var._index:
                raise IndexError("DifferentialSet '%s' is not an indexing set of the variable '%s'"\
                    % (ds.name,var.name))
        elif ds not in var._index_set:
            raise IndexError("DifferentialSet '%s' is not an indexing set of the variable '%s'"\
                % (ds.name,var.name))

        if self._reduced_cp.has_key(var.name):
            temp = self._reduced_cp[var.name]
            if temp.has_key(ds.name):
                raise RuntimeError("Variable '%s' has already been constrained to a reduced "\
                    "number of collocation points over DifferentialSet '%s'.")
            else:
                temp[ds.name]=ncp
        else:
            self._reduced_cp[var.name] = {ds.name:ncp}                
        
        list_name = var.name+"_interpolation_constraints"
        
        instance.add_component(list_name,ConstraintList(noruleinit=True))
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

