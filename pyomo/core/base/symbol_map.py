#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = [ 'BasicSymbolMap', 'SymbolMap', "CounterLabeler", 'NumericLabeler', 'TextLabeler',
            'NameLabeler', 'symbol_map_from_instance' ]

from weakref import ref as weakref_ref
from six import itervalues, iteritems, iterkeys
from pyomo.core.base import label_from_name
from pyomo.core.base.block import active_components_data

class CounterLabeler(object):

    def __init__(self, start=0):

        self._id = start

    def __call__(self, obj=None):

        self._id += 1
        return self._id

class NumericLabeler(object):

    def __init__(self, prefix, start=0):

        self.id = start
        self.prefix = prefix

    def __call__(self, obj=None):

        self.id += 1
        return self.prefix + str(self.id)

class TextLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return label_from_name(obj.cname(True, self.name_buffer))

# Same as above but without symbol translations needed for LP files
class NameLabeler(object):
    def __init__(self):
        self.name_buffer = {}

    def __call__(self, obj):
        return obj.cname(True, self.name_buffer)

# 
# an experimental utility method to create a symbol map from an
# instance. really will only work if name-based labelers are used, but
# that's good enough for now. also won't work with blocks.
#

def symbol_map_from_instance(instance):

   from pyomo.core.base import Var, Constraint, Objective

   resulting_map = SymbolMap(instance)

   labeler = TextLabeler()

   for block in instance.all_blocks(active=True):
      for varvalue in active_components_data(block,Var):
          # ignore the return value - we're just trying to populate the map.
          symbol = resulting_map.getSymbol(varvalue, labeler)

   for block in instance.all_blocks(active=True):
       for constraint_data in active_components_data(block,Constraint):
           con_symbol = resulting_map.getSymbol( constraint_data, labeler )               
           if constraint_data._equality:               
               label = 'c_e_' + con_symbol + '_'
               resulting_map.alias(constraint_data, label)
           else:
               if constraint_data.lower is not None:
                   if constraint_data.upper is not None:
                       resulting_map.alias(constraint_data, 'r_l_' + con_symbol + '_')
                       resulting_map.alias(constraint_data, 'r_u_' + con_symbol + '_')
                   else:
                       label = 'c_l_' + con_symbol + '_'
                       resulting_map.alias(constraint_data, label)
               elif constraint_data.upper is not None:
                   label = 'c_u_' + con_symbol + '_'
                   resulting_map.alias(constraint_data, label)

   for objective_data in active_components_data(instance,Objective):
       # ignore the return value - we're just trying to populate the map.
       resulting_map.getSymbol(objective_data, labeler)      
       resulting_map.alias(objective_data, "__default_objective__")

   return resulting_map


#
# a symbol map is a mechanism for tracking assigned labels (e.g., for
# use when writing problem files for input to an optimizer) for objects
# in a particular problem instance.
#
#
# This is a very simple implementation of symbol map. It does not support
# pickling nor does it support deleting the components it references. Nor
# does it support integration with the Labeler hierarchy - rather, callers
# supply their own labels. Further, aliasing functionality is not # provided. 
# It is up to the user to clean the symbol map in order to allow proper 
# garbage collection of deleted components.
#

class BasicSymbolMap(object):

    def __init__(self):
        
        # maps object id()s to their assigned symbol.
        self.byObject = {}
        
        # maps assigned symbols to the corresponding objects.
        self.bySymbol = {}

    def getByObjectDictionary(self):
        return self.byObject

    def updateSymbols(self, data_stream):
        self.byObject.update((id(obj), label) for obj,label in data_stream)
        self.bySymbol.update((label,obj) for obj,label in data_stream)

    def createSymbol(self, obj ,label):
        self.byObject[id(obj)] = label
        self.bySymbol[label] = obj        

    def updateSymbol(self, obj, label):
        self.byObject[id(obj)] = label
        self.bySymbol[label] = obj

    def getSymbol(self, obj):
        return self.byObject[id(obj)]

    def getObject(self, label):
        return self.bySymbol[label]

    def pprint(self, **kwds):
        print("BasicSymbolMap:")
        lines = [str(label)+" <-> "+obj.cname(True) for label, obj in iteritems(self.bySymbol)]
        print('\n'.join(sorted(lines)))
        print("")


#
# a symbol map is a mechanism for tracking assigned labels (e.g., for
# use when writing problem files for input to an optimizer) for objects
# in a particular problem instance.
#
# This version of symbol map supports pickling with the instance it
# references and has a number of features not found on the basic symbol
# map.
#
#

class SymbolMap(object):

    class UnknownSymbol:
        pass

    def __init__(self, instance):

        # conceptually, a symbol map must be associated with an instance
        # - for example, the byObject map creates associations between
        # object ids and their symbols => the ids are tied to a specific
        # instance. however, we don't actually do anything with the
        # instance within this class quite yet.
        self.instance = weakref_ref(instance)
        
        # maps object id()s to their assigned symbol.
        self.byObject = {}
        
        # maps assigned symbols to the corresponding objects.
        self.bySymbol = {}
        
        # maps alternative symbols to the corresponding objects
        self.aliases = {}

    def __getstate__(self):
        """
        This method must be defined for deepcopy/pickling because 
        this class relies on component ids.
        """
        result = dict()
        if self.instance is None:
            result['instance'] = None
        else:
            result['instance'] = self.instance()
        # The byObject dictionary will be rebuilt by using the 
        # bySymbol dictionary in __setstate__
        result['bySymbol'] = tuple((key,val()) for key,val in iteritems(self.bySymbol) if val() is not None)
        result['aliases'] = tuple((key,val()) for key,val in iteritems(self.aliases) if val() is not None)
        # take care of any other symbol_map components (if they are ever added)
        all_keys = set(iterkeys(self.__dict__))
        all_keys.difference_update(('bySymbol','byObject','aliases','instance'))
        for key in all_keys:
            result[key] = self.__dict__[key]
        return result

    def __setstate__(self, state):
        """
        This method must be defined for deepcopy/pickling because 
        this class relies on component ids.
        """
        if state['instance'] is None:
            self.instance = None
        else:
            self.instance = weakref_ref(state['instance'])
        self.byObject = dict((id(val),key) for key,val in state['bySymbol'])
        self.bySymbol = dict((key,weakref_ref(val)) for key,val in state['bySymbol'])
        self.aliases = dict((key,weakref_ref(val)) for key,val in state['aliases'])
        
    # 
    # it is often useful to extract the by-object dictionary
    # to directly access it, principally to avoid the overhead
    # associated with function calls and error checking - in
    # cases where you know an object will be in the dictionary.
    # this method is useful in cases such as the problem
    # writers, in which a single pass is performed through all
    # objects in the model to populate the symbol map - it 
    # is read-only after that point.
    #
    def getByObjectDictionary(self):
        return self.byObject

    #
    # invoked when the caller guarantees that a name conflict will not
    # arise and the use of a labeler is not required. use with care!
    # input is assumed to be an iterable of (object, label) pairs.
    # can alternatively be used when the caller wants to deal with 
    # labeling, and only invoke the symbol map update in bulk.
    #
    def updateSymbols(self, obj_symbol_tuples):
        self.byObject.update((id(obj), symb) for obj,symb in obj_symbol_tuples)
        self.bySymbol.update((symb, weakref_ref(obj)) for obj,symb in obj_symbol_tuples)

    #
    # invoked when the caller guarantees that a name conflict will not
    # arise. use with care!
    #
    def createSymbol(self, obj, labeler, *args):

        # the following test is slightly faster than always calling *args.
        if args:
            ans = labeler(obj, *args)
        else:
            ans = labeler(obj)
        self.byObject[id(obj)] = ans
        self.bySymbol[ans] = weakref_ref(obj)
        return ans

    #
    # same as above, but with a list (more generally, an iterable) of objs.
    #
    def createSymbols(self, objs, labeler, *args):

       byObjectDictionary = self.byObject
       bySymbolDictionary = self.bySymbol
       # the following test is slightly faster than always calling *args.
       if args:
           self.updateSymbols([(obj,labeler(obj, *args)) for obj in objs])
       else:
           self.updateSymbols([(obj,labeler(obj)) for obj in objs])
  
    #
    # same as above, but with full error checking for duplicates / collisions.
    #
    def getSymbol(self, obj, labeler, *args):

        obj_id = id(obj)
        if obj_id in self.byObject:
            # **NOTE: There is a risk here that the id can be invalid (i.e., reassigned
            #         after deletion of the original object. The following block of code
            #         could be used to protect against this case. It involves an extra
            #         lookup in the bySymbol dictionary to make sure the object weakref
            #         is still valid.
            ###symbol = self.byObject[obj_id]
            ###if obj is self.bySymbol[symbol]():
            ###    return symbol
            ###else:
            ###    del self.byObject[obj_id]
            ###    del self.bySymbol[symbol]
            return self.byObject[obj_id]
        
        # the following test is slightly faster than always calling *args
        if args:
            ans = labeler(obj, *args)
        else:
            ans = labeler(obj)
        
        if ans in self.bySymbol:
            if self.bySymbol[ans]() is not obj:
                raise RuntimeError(
                    "Duplicate symbol '%s' already associated with "
                    "component '%s' (conflicting component: '%s')"
                    % (ans, self.bySymbol[ans].cname(True), obj.cname(True)) )
        self.bySymbol[ans] = weakref_ref(obj)
        self.byObject[obj_id] = ans
        return ans

    def clearAliases(self):
        self.aliases = {}

    def alias(self, obj, name):

        #if id(obj) not in self.byObject:
        #    raise RuntimeError(
        #        "Cannot alias object '%s': object not in SymbolMap."
        #        % ( name, ))
        #if self.aliases.setdefault(name, obj) is not obj:
        if name in self.aliases:
            old_object = self.aliases[name]()
            if old_object is None:
                self.aliases[name] = weakref_ref(obj)
                return
            if old_object is not obj:
                raise RuntimeError(
                    "Duplicate alias '%s' already associated with "
                    "component '%s' (conflicting component: '%s')"
                    % (name, old_object.cname(True), obj.cname(True)) )
        self.aliases[name] = weakref_ref(obj)

    def getObject(self, symbol):
        if symbol in self.bySymbol:
            ans = self.bySymbol[symbol]()
            if not ans is None:
                return ans
            else:
                del self.bySymbol[symbol]
                return SymbolMap.UnknownSymbol
        elif symbol in self.aliases:
            ans = self.aliases[symbol]()
            if not ans is None:
                return ans
            else:
                del self.aliases[symbol]
                return SymbolMap.UnknownSymbol
        else: 
            return SymbolMap.UnknownSymbol 
            
    def getEquivalentObject(self, symbol, instance):

        obj = SymbolMap.UnknownSymbol
        if symbol in self.bySymbol:
            obj = self.bySymbol[symbol]()
            if obj is None:
                del self.bySymbol[symbol]
                obj = SymbolMap.UnknownSymbol
        if obj is SymbolMap.UnknownSymbol:
            if symbol in self.aliases:
                obj = self.aliases[symbol]()
                if obj is None:
                    del self.aliases[symbol]
                    return SymbolMap.UnknownSymbol
            else:
                return SymbolMap.UnknownSymbol
            
        path = []
        while obj is not None:
            if obj.parent_component() is obj:
                idx = None
            else:
                idx = obj.index()
            path.append((obj.parent_component(), idx))
            obj = obj.parent_block()
        path.pop() # skip the model itself
        newobj = instance
        for obj, idx in path:
            try:
                newobj = getattr(newobj, obj.cname())[idx]
            except AttributeError:
                return SymbolMap.UnknownSymbol
            except KeyError:
                return SymbolMap.UnknownSymbol
        return newobj

