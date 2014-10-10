#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import logging
import sys
import types
import pyutilib.math
from coopr.pyomo import *
from coopr.dae import *
from coopr.pyomo.base.sparse_indexed_component import *
from coopr.pyomo.base.misc import apply_indexed_rule

logger = logging.getLogger('coopr.pyomo')

def generate_finite_elements(ds,nfe):
    """ 
    This function first checks to see if the number of finite elements
    in the differential set is equal to nfe. If the number of finite
    elements is less than nfe, additional points will be generated. If
    the number of finite elements is greater than or equal to nfe the 
    differential set will not be modified
    """
    if (len(ds)-1) >= nfe:
        # In this case the differentialset already contains the
        # desired number or more than the desired number of finite
        # elements so no additional points are needed.
        return     
    elif len(ds) == 2:
        # If only bounds have been specified on the differentialset we
        # generate the desired number of finite elements by 
        # spreading them evenly over the interval
        step = (max(ds)-min(ds))/float(nfe)
        tmp = min(ds)+step
        while tmp<=(max(ds)-step):
            ds.add(tmp)
            tmp+=step
        ds.set_changed(True)
        ds._fe = sorted(ds)
        return
    else:
        # This is the case where some points have been specified
        # inside of the bounds however the desired number of finite 
        # elements has not been met. We first look at the step sizes
        # between the existing points. Then an additional point
        # is placed at the midpoint of the largest step. This
        # process is repeated until we have achieved the desired 
        # number of finite elements. If there are multiple "largest steps"
        # the point will be placed at the first occurance of the
        # largest step

        addpts = nfe-(len(ds)-1)
        while addpts>0:
            _add_point(ds)
            addpts -= 1
        ds.set_changed(True)
        ds._fe = sorted(ds)
        return

def _add_point(ds):
    sortds = sorted(ds)
    maxstep = sortds[1]-sortds[0]
    maxloc = 0
    for i in range(2,len(sortds)):
        if (sortds[i]-sortds[i-1])>maxstep:
            maxstep = sortds[i]-sortds[i-1]
            maxloc = i-1
            
    ds.add(sortds[maxloc]+maxstep/2.0)

def generate_colloc_points(ds,tau,radau):
    """
    This function adds collocation points between the finite elements
    in the differential set
    """
    if len(tau)==1 and radau:
        # Radau collocation has a collocation point at the finite
        # element locations so no need to add additional points
        return
    ds.set_changed(True)
    fes = sorted(ds)
    for i in range(1,len(fes)):
        h = fes[i]-fes[i-1]
        for j in range(len(tau)):
            pt = fes[i-1]+h*tau[j]
            if pt not in ds:
                ds.add(pt)

def update_diffset_indexed_component(comp):
    """
    Update any model components other than Differential
    which are indexed by a DifferentialSet that has changed
    """
    # FIXME: This implementation is a hack until Var and Constraint get
    # moved over to Sparse_Indexed_Component. The update methods below are 
    # roughly what the '_default' method in Var and Constraint should
    # do when they get reimplemented.

    # This implementation also assumes that only Var and Constraint components
    # will be explicitly indexed by a DifferentialSet and thus only checks for 
    # these two components. 

    # Additionally, this implemenation will *NOT* check
    # for or update components which use a DifferentialSet implicitly. ex) an objective
    # function which iterates through a DifferentialSet and sums the squared error. 
    # If you use a DifferentialSet implicitly you must initialize it with every
    # index you would like to have access to!

    if comp.type() is Suffix:
        return
    if comp.dim() == 1:
        if comp._index.type() == DifferentialSet: 
            if comp._index.get_changed():
                if comp.type() == Var:
                    _update_var(comp)
                elif comp.type() == Constraint:
                    _update_constraint(comp)
    elif comp.dim() > 1:

        if isinstance(comp,SparseIndexedComponent):
            indexset = comp._implicit_subsets
        else:
            indexset = comp._index_set

        for s in indexset:
            if s.type() == DifferentialSet and s.get_changed:
                if comp.type() == Var:
                    _update_var(comp)
                elif comp.type() == Constraint:
                    _update_constraint(comp)
                
                    
def _update_var(v):
    """
    This method will construct any additional indices in a variable 
    resulting from the discretization of DifferentialSet 
    """

    # Note: This is not required it is handled by the _default method on
    #       Var (which is now a SparseIndexedComponent). However, it
    #       would be much slower to rely on that method to generate new
    #       _VarData for a large number of new indices.
    new_indices = set(v._index)-set(v._data.keys())
    v._add_members(new_indices)
    v._initialize_members(new_indices)

def _update_constraint(con):
    """
    This method will construct any additional indices in a constraint
    resulting from the discretization.
    """

    # FIXME: This isn't quite as hackish as _update_constraint but it
    # would be nice to move this "update" operation to the contraint
    # object itself

    for i in con.index_set():
        if i not in con.keys():
            # Code taken from the construct() method of Constraint
            _rule=con.rule
            _parent=con._parent()
            con.add(i,apply_indexed_rule(con,_rule,_parent,i))

def add_equality_constraints(diff):
    """
    This function generates the equality constraints that set the 
    derivative at each index of the differential variable equal to
    the expression generated using the rule supplied for the rhs when 
    the Differential block was declared. This equality constraint should
    not be enforced at the initial point in the DifferentialSet.
    """

    # NOTE!!! The differential equation is currently not being
    # enforced at the first point in the differentialset (i.e. time zero).
    # This could lead to major issues if a user tries to specify an
    # initial condition for a derivative at time zero. If var was a
    # sparse_indexed_component we could deal with this
    ds = diff.get_differentialset()
    start = ds._fe[0]
    for i in diff.get_diffvar().keys():
        if isinstance(i,tuple):
            if i[diff._ds_argindex] == start:
                continue
            expr = diff._rule(diff._parent(),*i)
        else:
            if i == start:
                continue
            expr = diff._rule(diff._parent(),i)
        diff._cons.add(diff[i]==expr)

def get_index_information(var,ds):
    """
    This method will find the index location of the ds in the var, return
    a list of the non_ds indices and return a function that can be used to
    access specific indices in var indexed by a DifferentialSet by specifying the
    finite elemtent and collocation point. Users of this method should have
    already confirmed that ds is an indexing set of var and that it's a DifferentialSet
    """

    # Find index order of differentialset in the variable
    indargs=[]
    dsindex=0
    tmpds2=None

    if var.dim() != 1:
        # If/when Var is changed to SparseIndexedComponent, the _index_set 
        # attribute below may need to be changed
        indCount = 0
        for index in var._implicit_subsets:
            if isinstance(index,DifferentialSet):
                if index ==ds:
                    dsindex = indCount
                else:
                    indargs.append(index)  # If var is indexed by multiple differentialsets treat
                                           # other differentialsets like a normal idexing set
                indCount += 1     # A differentialset must be one dimensional
            else:
                indargs.append(index)
                indCount += index.dimen

    if indargs == []:
        non_ds = (None,)
    elif len(indargs)>1:
        non_ds = tuple(a for a in indargs)
    else:
        non_ds = (indargs[0],)

    if None in non_ds:
        tmpidx = (None,)
    elif len(non_ds)==1:
        tmpidx = non_ds[0]
    else:
        tmpidx = non_ds[0].cross(*non_ds[1:])

    # Lambda function used to generate the desired index
    # more concisely
    idx = lambda n,i,k: _get_idx(dsindex,ds,n,i,k)

    info = {}
    info['non_ds']=tmpidx
    info['index function'] = idx
    return info


def _get_idx(l,ds,n,i,k):
    """
    This function returns the appropriate index for a variable 
    indexed by a differential set. It's needed because the collocation 
    constraints are indexed by finite element and collocation point
    however a differentialset contains a list of all the discretization
    points and is not separated into finite elements and collocation
    points.
    """
    t = sorted(ds)
    tmp = t.index(ds._fe[i])
    tik = t[tmp+k]
    if n is None:
        return tik
    else:
        tmpn=n
        if not isinstance(n,tuple):
            tmpn = (n,)
    return tmpn[0:l]+(tik,)+tmpn[l:]



