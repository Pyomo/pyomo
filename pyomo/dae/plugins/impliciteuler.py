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

from coopr.core.plugin import alias
from coopr.pyomo.base import Transformation
from coopr.pyomo import *
from coopr.pyomo.base import Block
from coopr.dae import *
from coopr.dae.misc import generate_finite_elements
from coopr.dae.misc import update_diffset_indexed_component
from coopr.dae.misc import add_equality_constraints


logger = logging.getLogger('coopr.pyomo')


class Implicit_Euler_Transformation(Transformation):

    alias('dae.euler_discretization', doc="TODO")

    def __init__(self):
        super(Implicit_Euler_Transformation, self).__init__()

    def _setup(self, instance):
        instance = instance.clone()
        instance.concrete_mode()
        self._nfe = {}
        self._blocks = list(instance.all_blocks())
        return instance

    def apply(self, instance, **kwds):
        instance = self._setup(instance)  
        tmpnfe = kwds.pop('nfe',10)
        tmpds = kwds.pop('diffset',None)

        if tmpds is not None and tmpds.type() is not DifferentialSet:
            raise TypeError("The component specified using the 'diffset' keyword "\
                "must be a differential set")

        if self._nfe.has_key(None):
            raise ValueError("A general discretization scheme has already been applied to "\
                    "to every differential set in the model. If you would like to specify a "\
                    "specific discretization scheme for one of the differential sets you must discretize "\
                    "each differential set individually. If you would like to apply a different "\
                    "discretization scheme to all differential sets you must declare a new Implicit_"\
                    "Euler object")

        if len(self._nfe) == 0 and tmpds is None:
            # Same discretization on all differentialsets
            self._nfe[None] = tmpnfe
            currentds = None
        elif self._nfe.has_key(tmpds.name):
            raise ValueError("A discretization scheme has already been specified "\
                    "for DifferentialSet '%s'" %s(tmpds.name))
        else :
            self._nfe[tmpds.name]=tmpnfe
            currentds = tmpds.name

        # Note: all_blocks is a generator and since we are changing the
        # type of Differential components to Block after the discretization
        # we have to explicitly form the list of all Blocks before iterating
        # over them. Otherwise, the differential components will be included
        # in the iteration and _transformBlock will be called unintentionally 
        # for each differential component after their type has been changed.
        for block in self._blocks:
            print(block)
            self._transformBlock(block,currentds)

        # Taken from bigm
        # REQUIRED: re-call preprocess()
        instance.preprocess()
        return instance

    def _transformBlock(self, block,currentds):

        self._fe = {}
        for ds in block.components(DifferentialSet).itervalues():
            if currentds is None or currentds is ds.name:
                generate_finite_elements(ds,self._nfe[currentds])
                if not ds.get_changed():
                    if len(ds)-1 > self._nfe[currentds]:
                        print("***WARNING: More finite elements were found in DifferentialSet "\
                            "'%s' than the number of finite elements specified in apply. "\
                            "The larger number of finite elements will be used." %s(ds.name))
                
                self._nfe[ds.name]=len(ds)-1
                self._fe[ds.name]=sorted(ds)
                ds._fe=sorted(ds)
                # Adding discretization information to the differentialset object itself
                # so that it can be accessed outside of the discretization object
                disc_info = ds.get_discretization_info()
                disc_info['nfe']=self._nfe[ds.name]
                disc_info['scheme']='Implicit Euler'
                     
        # Maybe check to see if any of the DifferentialSets have been changed,
        # if they haven't then the model components need not be updated
        # or even iterated through

        for c in block.components().itervalues():
            if c.type() != Differential:
                update_diffset_indexed_component(c)
            else:
                update_diffset_indexed_component(c._lhs_var)
        
        for d in block.components(Differential).itervalues():
            if currentds is None or d.get_differentialset().name is currentds:
                add_equality_constraints(d)
                self._impEuler_transform(d,)
                block.reclassify_component_type(d,Block)
            
    def _impEuler_transform(self,diff):
        # generate the constraints unique to this discretization scheme
        # The needed constraint is:
        # 1) y[t_k+1] == y[t_k] + (t_k+1 - t_k)*y_dot[t_k+1]
        t = sorted(diff.get_differentialset())
        l = diff._ds_argindex
        dv = diff.get_diffvar()
        if None in diff._non_ds:
            tmpidx = (None,)
        elif len(diff._non_ds)==1:
            tmpidx = diff._non_ds[0]
        else:
            tmpidx = diff._non_ds[0].cross(*diff._non_ds[1:])

        for i in tmpidx:
            for k in range(len(t)-1):
                if i is None:
                    idx1 = t[k]
                    idx2 = t[k+1]
                else:
                    tmpi=i
                    if not isinstance(i,tuple):
                        tmpi = (i,)
                    idx1 = tmpi[0:l]+(t[k],)+tmpi[l:]
                    idx2 = tmpi[0:l]+(t[k+1],)+tmpi[l:]

                diff._cons.add(dv[idx2]==dv[idx1]+(t[k+1]-t[k])*diff[idx2])
