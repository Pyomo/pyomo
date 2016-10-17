#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyutilib.misc.indent_io import StreamIndenter

from pyomo.core import *
from pyomo.core.base.constraint import (SimpleConstraint,
                                        IndexedConstraint,
                                        _GeneralConstraintData)
from pyomo.core.base.block import _BlockData
from pyomo.core.base.sets import Set
from pyomo.core.base.indexed_component import normalize_index

from os.path import abspath, dirname, join, normpath
pyomo_base = normpath(join(dirname(abspath(__file__)), '..', '..', '..'))

from pyutilib.misc import LogHandler

import logging
logger = logging.getLogger('pyomo.gdp')
logger.setLevel(logging.WARNING)
logger.addHandler( LogHandler(
    pyomo_base, verbosity=lambda: logger.isEnabledFor(logging.DEBUG)) )


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
        ostream.write("    %s :" % self.local_name)
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
            logger.warning("Discarding pre-defined M values for %s", self.name)
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
        ## where _BlockData creates the indicator_var *before*
        ## Block.__init__ declares the _defer_construction flag.
        self._defer_construction = True
        self._suppress_ctypes = set()

        _DisjunctData.__init__(self, self)
        Disjunct.__init__(self, *args, **kwds)
        self._data[None] = self

    def pprint(self, ostream=None, verbose=False, prefix=""):
        Disjunct.pprint(self, ostream=ostream, verbose=verbose, prefix=prefix)


class IndexedDisjunct(Disjunct):
    pass

class Disjunction(Constraint):

    def __new__(cls, *args, **kwds):
        if cls != Disjunction:
            return super(Disjunction, cls).__new__(cls)
        if args == ():
            return SimpleDisjunction.__new__(SimpleDisjunction)
        else:
            return IndexedDisjunction.__new__(IndexedDisjunction)

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

class SimpleDisjunction(_GeneralConstraintData, Disjunction):

    def __init__(self, *args, **kwds):
        _GeneralConstraintData.__init__(self, None, component=self)
        Disjunction.__init__(self, *args, **kwds)

    #
    # Since this class derives from Component and
    # Component.__getstate__ just packs up the entire __dict__ into
    # the state dict, we do not need to define the __getstate__ or
    # __setstate__ methods.  We just defer to the super() get/set
    # state.  Since all of our get/set state methods rely on super()
    # to traverse the MRO, this will automatically pick up both the
    # Component and Data base classes.
    #

    #
    # Override abstract interface methods to first check for
    # construction
    #

    @property
    def body(self):
        """Access the body of a constraint expression."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the body of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.body.fget(self)
        raise ValueError(
            "Accessing the body of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the lower bound of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.lower.fget(self)
        raise ValueError(
            "Accessing the lower bound of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the upper bound of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.upper.fget(self)
        raise ValueError(
            "Accessing the upper bound of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the equality flag of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.equality.fget(self)
        raise ValueError(
            "Accessing the equality flag of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the strict_lower flag of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.strict_lower.fget(self)
        raise ValueError(
            "Accessing the strict_lower flag of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the strict_upper flag of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.strict_upper.fget(self)
        raise ValueError(
            "Accessing the strict_upper flag of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    #
    # Singleton constraints are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Constraint.Skip are managed. But after that they will behave
    # like _ConstraintData objects where set_value does not handle
    # Constraint.Skip but expects a valid expression or None.
    #

    def set_value(self, expr):
        """Set the expression on this constraint."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _GeneralConstraintData.set_value(self, expr)
        raise ValueError(
            "Setting the value of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no object to set)."
            % (self.name))

class IndexedDisjunction(Disjunction):
    pass

class _disjunctiveRuleMapper(object):
    def __init__(self, disjunction):
        self.disj = disjunction

    def __call__(self, *args):
        model = args[0]
        idx = args[1:]
        if len(idx)>1 and idx not in self.disj._index:
            logger.warning("Constructing disjunction from "
                           "unrecognized index: %s", str(idx))
        elif len(idx) == 1 and idx[0] not in self.disj._index:
            logger.warning("Constructing disjunction from "
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
                    (type(d), self.disj.name)
                raise ValueError(msg)

        self.disj._disjuncts[normalize_index(idx)] = disjuncts
        # sum(disjuncts) == 1
        return (sum(d.indicator_var for d in disjuncts), 1.0)


register_component(Disjunct, "Disjunctive blocks.")
register_component(Disjunction, "Disjunction expressions.")
