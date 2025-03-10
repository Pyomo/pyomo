#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


class SymbolMap(object):
    """
    A class for tracking assigned labels for modeling components.

    Symbol maps are used, for example, when writing problem files for
    input to an optimizer.

    Warning:
        A symbol map should never be pickled.  This class is
        typically constructed by solvers and writers, and it may be
        owned by models.

    Note:
        We should change the API to not use camelcase.

    Attributes
    ----------
    byObject : dict
        maps (object id) to (string label)
    bySymbol : dict
        maps (string label) to (object)
    aliases : dict
        maps (string label) to (object)
    default_labeler:
        used to compute a string label from an object
    """

    def __init__(self, labeler=None):
        self.byObject = {}
        self.bySymbol = {}
        self.aliases = {}
        self.default_labeler = labeler

    class UnknownSymbol:
        pass

    def __getstate__(self):
        #
        # While we should generally not pickle a SymbolMap, we still
        # need to implement __getstate__ / __setstate__ so that the
        # bi-map is correctly duplicated if the object is ever
        # deepcopied (the id() keys need to be updated to point to the
        # new model objects)
        #
        # Note: byObject and bySymbol constitute a bimap.  We only need
        # to save one of them.
        #
        return (self.bySymbol, self.aliases, self.default_labeler)

    def __setstate__(self, state):
        self.bySymbol, self.aliases, self.default_labeler = state
        self.byObject = {id(v): k for k, v in self.bySymbol.items()}

    def addSymbol(self, obj, symb):
        """
        Add a symbol for a given object

        This method assumes that objects and symbol names will not conflict.
        """
        nSymbols = len(self.byObject) + 1
        self.byObject[id(obj)] = symb
        self.bySymbol[symb] = obj
        if nSymbols != len(self.bySymbol):
            raise RuntimeError(
                "SymbolMap.addSymbol(): duplicate symbol.  "
                "SymbolMap likely in an inconsistent state"
            )
        if len(self.byObject) != len(self.bySymbol):
            raise RuntimeError(
                "SymbolMap.addSymbol(): duplicate object.  "
                "SymbolMap likely in an inconsistent state"
            )

    def addSymbols(self, obj_symbol_tuples):
        """
        Add (object, symbol) tuples from an iterable object.

        This method assumes that objects and symbol names will not conflict.
        """
        nSymbols = len(self.bySymbol)
        for obj, symbol in obj_symbol_tuples:
            self.byObject[id(obj)] = symbol
            self.bySymbol[symbol] = obj
            nSymbols += 1
        if nSymbols != len(self.bySymbol):
            raise RuntimeError(
                "SymbolMap.addSymbols(): duplicate symbol.  "
                "SymbolMap likely in an inconsistent state"
            )
        if len(self.byObject) != len(self.bySymbol):
            raise RuntimeError(
                "SymbolMap.addSymbols(): duplicate object.  "
                "SymbolMap likely in an inconsistent state"
            )

    def createSymbol(self, obj, labeler=None, *args):
        """
        Create a symbol for an object with a given labeler.  No
        error checking is done to ensure that the generated symbol
        name is unique.
        """
        if labeler is None:
            if self.default_labeler is not None:
                labeler = self.default_labeler
            else:
                labeler = str
        symbol = labeler(obj, *args)
        self.addSymbol(obj, symbol)
        return symbol

    def createSymbols(self, objs, labeler=None, *args):
        """
        Create a symbol for iterable objects with a given labeler.  No
        error checking is done to ensure that the generated symbol
        names are unique.
        """
        if labeler is None:
            if self.default_labeler is not None:
                labeler = self.default_labeler
            else:
                labeler = str
        self.addSymbols((obj, labeler(obj, *args)) for obj in objs)

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
        symbol = (labeler or self.default_labeler or str)(obj, *args)
        if symbol in self.bySymbol:
            # The labeler can have side-effects, including registering
            # this symbol in the symbol map
            if obj is self.bySymbol[symbol]:
                return symbol
            raise RuntimeError(
                "Duplicate symbol '%s' already associated with "
                "component '%s' (conflicting component: '%s')"
                % (symbol, self.bySymbol[symbol].name, obj.name)
            )
        self.bySymbol[symbol] = obj
        self.byObject[obj_id] = symbol
        return symbol

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
            old_object = self.aliases[name]
            if old_object is obj:
                return
            else:
                raise RuntimeError(
                    "Duplicate alias '%s' already associated with "
                    "component '%s' (conflicting component: '%s')"
                    % (
                        name,
                        "UNKNOWN" if old_object is None else old_object.name,
                        obj.name,
                    )
                )
        else:
            #
            # Add the alias
            #
            self.aliases[name] = obj

    def getObject(self, symbol):
        """
        Return the object corresponding to a symbol
        """
        if symbol in self.bySymbol:
            return self.bySymbol[symbol]
        elif symbol in self.aliases:
            return self.aliases[symbol]
        else:
            return SymbolMap.UnknownSymbol

    def removeSymbol(self, obj):
        symb = self.byObject.pop(id(obj))
        self.bySymbol.pop(symb)
