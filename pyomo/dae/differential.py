#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['Differential']

import sys
import types
import weakref
import logging

from pyomo.core import *
from pyomo.core.base.sets import *
from pyomo.core.base.misc import create_name, apply_indexed_rule
from pyomo.dae import DifferentialSet
from pyomo.core.base.block import SimpleBlock
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var

logger = logging.getLogger('pyomo.core')


class Differential(SimpleBlock):
    """
    An object that defines a differential equation
    
    keyword arguments:
    name: name of this object
    rule: a function or expression describing the right hand side
          of the differential equation
    expr: same as rule
    doc: documentation string for the differential equation
    dv: The variable being differentiated
    dvar: same as dv
    ds: The Differential Set over which the derivative of the differential
          variable is taken
    dset: same as ds
    bounds: A tuple or function setting bounds on the derivative of the
          differential variable

    Class Attributes:
        _ds_argindex: An integer representing where the differentialset index is located 
             in the order of all the indices
        _non_ds: The indexing sets that are not the differential set
        _lhs_var: Variable representing the derivative of the differential variable
        _cons: Constraints that are added during discretization
  
    """
  
    def __init__(self, *args, **kwd):
        if args != ():
            raise ValueError("A Differential should not have any positional arguments, "\
                                 "the indexing of a Differential is exactly the indexing "\
                                 "of the differential variable")     

        if "rule" in kwd and "expr" in kwd:
            raise TypeError("Cannot specify both 'rule' and 'expr' " + \
                "keywords in a Differential")
        if "dv" in kwd and "dvar" in kwd:
            raise TypeError("Cannot specify both 'dv' and 'diffvar' " +\
                "keywords in a Differential")
        if "ds" in kwd and "dset" in kwd:
            raise TypeError("Cannot specify both 'ds' and 'differentialset' " +\
                "keywords in a Differential")
        
        tmprule = kwd.pop("rule", None)
        tmprule = kwd.pop("expr", tmprule)
        if tmprule is None:
            raise TypeError("A rule must be specified for a Differential")
        elif not isinstance(tmprule,types.FunctionType):
            # Because a Differential must be indexed by at least a DifferentialSet the rule
            # supplied must be a function
            raise TypeError("A Differential rule must be a function")

        tmpdv = kwd.pop("dv", None)
        tmpdv = kwd.pop("dvar", tmpdv)
        if tmpdv is None:
            raise TypeError("A differential variable must be specified")
        elif not isinstance(tmpdv,Var):
            raise TypeError("A differential variable must be a variable")
        tmpbounds = kwd.pop("bounds",None)
        tmpinitial = kwd.pop("initialize",None)
        
        tmpds = kwd.pop("ds",None)
        tmpds = kwd.pop("dset",tmpds)
        if tmpds is not None:
            if not isinstance(tmpds,DifferentialSet):
                raise TypeError("The component specified as a differentialset is not "\
                    "a differentialset")

        # Check to make sure that a differential set specified by a keyword 
        # argument belongs to the indexing sets of the differential variable
        if not tmpds is None:
            if tmpdv.dim() == 1:
                if not tmpds in tmpdv.index_set():
                    raise TypeError("DifferentialSet '%s' is not an indexing set of the differential variable"\
                        % str(tmpds))
            elif not tmpds in tmpdv._implicit_subsets:
                raise TypeError("DifferentialSet '%s' is not an indexing set of the differential variable"\
                    % str(tmpds))
        
        # Find the index of the DifferentialSet in the differential variable. If
        # the differential variable is indexed by mulitple DifferentialSets check 
        # to make sure the DifferentialSet was set using a keyword argument. Also,
        # generate an indexing argument list which contains all the indicies
        # of the differential variable except for the differential set associated 
        # with this differential. DifferentialSets must be *Explicit* indexing sets
        # of the differential variable

        indargs=[]
        dsCount=0
        dsindex=0
        tmpds2=None

        if tmpdv.dim() == 1:
            if isinstance(tmpdv._index,DifferentialSet):
                dsCount = 1
                if tmpds is None:
                    tmpds = tmpdv._index
            else:
                raise IndexError("A differential variable must be indexed by a differential set")
        else:
            # If/when Var is changed to SparseIndexedComponent, the _index 
            # attribute below may need to be changed
            indCount = 0
            for index in tmpdv._implicit_subsets:
                if isinstance(index,DifferentialSet):
                    if not tmpds is None:
                        if index == tmpds:
                            dsindex = indCount
                        else:
                            indargs.append(index)  # If dv is indexed by multiple differentialsets treat
                                                   # other differentialsets like a normal idexing set
                    else:
                        tmpds2=index
                        dsindex=indCount
                    indCount += 1     # A differentialset must be one dimensional
                    dsCount += 1
                else:
                    indargs.append(index)
                    indCount += index.dimen
        
        if dsCount == 0:
            raise IndexError("Differential variable must be indexed by a DifferentialSet")
        elif dsCount == 1:
            if tmpds is None:
                tmpds = tmpds2
        else:
            if tmpds is None:
                raise TypeError("If a differential variable is indexed by multiple differentialsets "\
                    "the desired differentialset must be specified using a keyword argument")
        
        kwd.setdefault('ctype', Differential)
        
        SimpleBlock.__init__(self,**kwd)
     
        self._rule = tmprule
        self._bounds = tmpbounds
        self._initial = tmpinitial
        self._ds_argindex = dsindex
             
        # __setattr__ in block handles things derived from
        # component strangely. If we use it to add the three 
        # attributes below it will actually change their names
        # which isn't the desired behavior. This is following
        # something similar in Piecewise
        self.__dict__['_dv'] = tmpdv
        self.__dict__['_ds'] = tmpds


        if indargs == []:
            self.__dict__['_non_ds'] = (None,)
        elif len(indargs)>1:
            self.__dict__['_non_ds']=tuple(a for a in indargs)
        else:
            self.__dict__['_non_ds']=(indargs[0],)


    def __str__(self):
        """
        Return a string representation of the differential
        """
        if self.name is None:
            return ""
        else:
            return self.name
    
    def construct(self, data=None):
        """
        Creates derivative variables for the indices in the DifferentialSet which 
        have been specified before the discretization
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing differential '%s'",self.name)

        if self._constructed:
            return
        self._constructed = True
        self._defer_construction = False

        # Not sure if the following line is needed
        #self.concrete_mode()

        if type(self._initial) is types.FunctionType:
            # If a function is supplied to 'initialize' we modify it slightly
            # to ensure that the parent block for the differential gets passed
            # to the function and not the differential itself
            def _init(d,*args):
                return self._initial(self._parent(),*args)
            init=_init
        else:
            init=self._initial

        if type(self._bounds) is types.FunctionType:
            # If a function is supplied to 'bounds' we modify it slightly 
            # to ensure that the parent block for the differential gets passed 
            # to the function and not the differential itself
            def _dbounds(d,*args):
                return self._bounds(self._parent(),*args)
            bds = _dbounds
        else:
            bds = self._bounds
        
        # TODO: This will need to be changed when Var is moved to SparseIndexedComponent
        if self._dv.dim()==1: 
            self.add_component('_lhs_var',Var(self._dv._index,bounds=bds,initialize=init))
        else:
            self.add_component('_lhs_var',Var(*self._dv._implicit_subsets,bounds=bds,initialize=init))

        self.add_component('_cons',ConstraintList(noruleinit=True))
            
    def __getitem__(self,indx):

        if indx is None:
            return self
        if indx in self._lhs_var.keys():
            return self._lhs_var[indx]
        else:
            raise KeyError("Unknown index in Differential '%s': %s" % (self.name,str(indx)))

    def get_diffvar(self):
        return self._dv

    def get_differentialset(self):
        return self._ds

    def pprint(self, ostream=None, verbose=False, prefix=""):
        # TBD: Do something with the prefix input argument!
        if ostream is None:
            ostream = sys.stdout
        ostream.write("  %s : " %(self.name,))
        ostream.write("DiffVar=%s \tDifferentialSet=%s\n" %(self._dv.name,self._ds.name))
        if not self._constructed:
            ostream.write("\tNot Constructed\n")
            return
        else:
            self._lhs_var.pprint()
            self._cons.pprint()

register_component(Differential, "Differential equation expressions.")
