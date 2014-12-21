#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *
from pyomo.core.base.block import _BlockData
from pyomo.core.base.misc import apply_parameterized_indexed_rule
from pyomo.core.base.sets import Set
from pyomo.core.base.sparse_indexed_component import normalize_index
from pyutilib.misc.indent_io import StreamIndenter

import logging
logger = logging.getLogger('pyomo.core')


class GDP_Error(Exception):
    """Exception raised while processing GDP Models"""


class _DisjunctData(_BlockData):

    def __init__(self, owner):
        _BlockData.__init__(self, owner)
        self._M = None
        self.indicator_var = Var(within=Binary)

    def pprint(self, ostream=None, verbose=False, prefix=""):
        _BlockData.pprint(self, ostream=ostream, verbose=verbose, prefix=prefix)

    def Xpprint(self, ostream=None, verbose=False, prefix=""):
        if ostream is None:
            ostream = sys.stdout
        ostream.write("    %s :" % self.name)
        ostream.write("\tSize= %s" % str(len(self._data.keys())))
        if isinstance(self._index,Set):
            ostream.write("\tIndex= %s\n" % self._index.name)
        else:
            ostream.write("\n")
        if self._M is not None:
            ostream.write("\tM= %s\n" % str(self._M))
        indent = StreamIndenter(ostream)
        if len(self._data):
            for val in self._data.itervalues():
                val.pprint(indent, verbose=verbose, prefix=prefix)
        else:
            Block.pprint(self, ostream=indent, verbose=verbose, prefix=prefix)

    def next_M(self):
        if self._M is None:
            return None
        elif isinstance(self._M, list):
            if len(self._M):
                return self._M.pop(0)
            else:
                return None
        else:
            return self._M

    def add_M(self, M):
        if self._M is None:
            self._M = M
        elif isinstance(self._M, list):
            self._M.append(M)
        else:
            self._M = [self._M, M]

    def set_M(self, M_list):
        if self._M is not None:
            logger.warn("Discarding pre-defined M values for %s", self.name)
        self._M = M_list


class Disjunct(Block):

    def __new__(cls, *args, **kwds):
        if cls != Disjunct:
            return super(Disjunct, cls).__new__(cls)
        if args == ():
            return SimpleDisjunct.__new__(SimpleDisjunct)
        else:
            return IndexedDisjunct.__new__(IndexedDisjunct)
    
    def __init__(self, *args, **kwargs):
        if kwargs.pop('_deep_copying', None):
            # Hack for Python 2.4 compatibility
            # Deep copy will copy all items as necessary, so no need to
            # complete parsing
            return

        kwargs.setdefault('ctype', Disjunct)
        Block.__init__(self, *args, **kwargs)

    def _default(self, idx):
        return self._data.setdefault(idx, _DisjunctData(self))

  
class SimpleDisjunct(_DisjunctData, Disjunct):

    def __init__(self, *args, **kwds):
        ## FIXME: This is a HACK to get around a chicken-and-egg issue
        ## where _BloclData creates the indicator_var *before*
        ## Block.__init__ declares the _defer_construction flag.
        self._defer_construction = True
        self._suppress_ctypes = set()

        _DisjunctData.__init__(self, self)
        Disjunct.__init__(self, *args, **kwds)
        self._data[None] = self

    def pprint(self, ostream=None, verbose=False, prefix=""):
        Disjunct.pprint(self, ostream, verbose, prefix)


class IndexedDisjunct(Disjunct):

    def __init__(self, *args, **kwds):
        Disjunct.__init__(self, *args, **kwds)



class Disjunction(Constraint):

    def __init__(self, *args, **kwargs):
        tmpname = kwargs.get('name', 'unknown')
        self._disjunction_rule = kwargs.pop('rule', None)
        self._disjunctive_set = kwargs.pop('expr', None)
        self._disjuncts = {}

        kwargs.setdefault('ctype', Disjunction)
        kwargs['rule'] = _disjunctiveRuleMapper(self)
        Constraint.__init__(self, *args, **kwargs)

        if self._disjunction_rule is not None and \
                self._disjunctive_set is not None:
            msg = "Creating disjunction '%s' that specified both rule " \
                "and expression" % tmpname
            raise ValueError(msg)
        #if ( kwargs ): # if dict not empty, there's an error.  Let user know.
        #    msg = "Creating disjunction '%s': unknown option(s)\n\t%s"
        #    msg = msg % ( tmpname, ', '.join(kwargs.keys()) )
        #    raise ValueError(msg)


class _disjunctiveRuleMapper(object):
    def __init__(self, disjunction):
        self.disj = disjunction

    def __call__(self, *args):
        model = args[0]
        idx = args[1:]
        if len(idx)>1 and idx not in self.disj._index:
            logger.warn("Constructing disjunction from "
                        "unrecognized index: %s", str(idx))
        elif len(idx) == 1 and idx[0] not in self.disj._index:
            logger.warn("Constructing disjunction from "
                        "unrecognized index: %s", str(idx[0]))
        elif not idx:
            idx = None

        if self.disj._disjunction_rule is not None:
            tmp = list(args)
            #tmp.append(self.disj)
            tmp = tuple(tmp)
            disjuncts = self.disj._disjunction_rule(*tmp)
        elif type(self.disj._disjunctive_set) in (tuple, list):
            # explicit disjunction over a user-specified list of disjuncts
            disjuncts = self.disj._disjunctive_set
        elif isinstance(self.disj._disjunctive_set, Disjunct):
            # pick one of all disjuncts
            if len(self.disj._disjunctive_set._data):
                disjuncts = self.disj._data.values()
            else:
                disjuncts = ( self.disj._disjunctive_set )
        else:
            msg = 'Bad expression passed to Disjunction'
            raise TypeError(msg)

        for d in disjuncts:
            if not isinstance(d, _DisjunctData):
                msg = 'Non-disjunct (type="%s") found in ' \
                    'disjunctive set for disjunction %s' % \
                    ( type(d), self.disj.name )
                raise ValueError(msg)

        self.disj._disjuncts[normalize_index(idx)] = disjuncts
        # sum(disjuncts) == 1
        return (sum(d.indicator_var for d in disjuncts), 1.0)


register_component(Disjunct, "Disjunctive blocks.")
register_component(Disjunction, "Disjunction expressions.")
