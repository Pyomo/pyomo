#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from weakref import ref as weakref_ref

from six import iteritems, iterkeys

#
# A symbol map is a mechanism for tracking assigned labels (e.g., for
# use when writing problem files for input to an optimizer) for objects
# in a particular problem instance.
#
# A symbol map should never be pickled.  This class is constructed
# by solvers and writers, and it owned by models.
#
class SymbolMap(object):

    class UnknownSymbol:
        pass

    def __init__(self):
        #
        # byObject:  id -> string
        # bySymbol:  string -> object weakref
        # alias:     string -> object weakref
        #
        self.byObject = {}
        self.bySymbol = {}
        self.aliases = {}

    def __getstate__(self):
        # Note: byObject and bySymbol constitute a bimap.  We only need
        # to pickle one of them, and bySymbol is easier.
        return {
            'bySymbol': tuple(
                (key, obj()) for key, obj in iteritems(self.bySymbol) ),
            'aliases': tuple(
                (key, obj()) for key, obj in iteritems(self.aliases) ),
        }

    def __setstate__(self, state):
        self.byObject = dict(
            (id(obj), key) for key, obj  in state['bySymbol'] )
        self.bySymbol = dict(
            (key, weakref_ref(obj)) for key, obj in state['bySymbol'] )
        self.aliases = dict(
            (key, weakref_ref(obj)) for key, obj in state['aliases'] )

    def addSymbol(self, obj, symb):
        self.byObject[id(obj)] = symb
        self.bySymbol[symb] = weakref_ref(obj)

    def addSymbols(self, obj_symbol_tuples):
        """
        Add (object, symbol) tuples from an iterable object.

        This method assumes that symbol names will not conflict.
        """
        tuples = list((obj, symb) for obj,symb in obj_symbol_tuples)
        self.byObject.update((id(obj_), symb_) for obj_,symb_ in tuples)
        self.bySymbol.update((symb_, weakref_ref(obj_)) for obj_,symb_ in tuples)

    def createSymbol(self, obj, labeler, *args):
        """
        Create a symbol for an object with a given labeler.  No
        error checking is done to ensure that the generated symbol
        name is unique.
        """
        #if args:
        #    symb = labeler(obj, *args)
        #else:
        #    symb = labeler(obj)
        symb = labeler(obj)
        self.byObject[id(obj)] = symb
        self.bySymbol[symb] = weakref_ref(obj)
        return symb

    def createSymbols(self, objs, labeler, *args):
        """
        Create a symbol for iterable objects with a given labeler.  No
        error checking is done to ensure that the generated symbol
        names are unique.
        """
        #if args:
        #    self.addSymbols([(obj,labeler(obj, *args)) for obj in objs])
        #else:
        #    self.addSymbols([(obj,labeler(obj)) for obj in objs])
        self.addSymbols([(obj,labeler(obj)) for obj in objs])

    def getSymbol(self, obj, labeler=None, *args):
        """
        Return the symbol for an object.  If it has not already been cached
        in the symbol map, then create it.
        """
        obj_id = id(obj)
        if obj_id in self.byObject:
            return self.byObject[obj_id]
        #
        # Create a new symbol, performing an error check if it is a duplicate
        #
        if labeler is None:
            raise RuntimeError("Object %s is not in the symbol map. "
                               "Cannot create a new symbol without "
                               "a labeler." % obj.name)
        symb = labeler(obj)
        if symb in self.bySymbol:
            if self.bySymbol[symb]() is not obj:
                raise RuntimeError(
                    "Duplicate symbol '%s' already associated with "
                    "component '%s' (conflicting component: '%s')"
                    % (symb, self.bySymbol[symb]().name, obj.name) )
        self.bySymbol[symb] = weakref_ref(obj)
        self.byObject[obj_id] = symb
        return symb

    def alias(self, obj, name):
        """
        Create an alias for an object.  An aliases are symbols that
        do not have a one-to-one correspondence with objects.
        """
        if name in self.aliases:
            #
            # If the alias exists and the objects are the same,
            # then return.  Otherwise, raise an exception.
            #
            old_object = self.aliases[name]()
            if old_object is obj:
                return
            else:
                raise RuntimeError(
                    "Duplicate alias '%s' already associated with "
                    "component '%s' (conflicting component: '%s')"
                    % (name, "UNKNOWN" if old_object is None else old_object.name, obj.name) )
        else:
            #
            # Add the alias
            #
            self.aliases[name] = weakref_ref(obj)

    def getObject(self, symbol):
        """
        Return the object corresponding to a symbol
        """
        if symbol in self.bySymbol:
            return self.bySymbol[symbol]()
        elif symbol in self.aliases:
            return self.aliases[symbol]()
        else:
            return SymbolMap.UnknownSymbol

    def removeSymbol(self, obj):
        symb = self.byObject.pop(id(obj))
        self.bySymbol.pop(symb)
