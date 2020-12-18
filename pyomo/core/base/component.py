#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import six
import sys
from copy import deepcopy
from pickle import PickleError
from six import iteritems, string_types
from weakref import ref as weakref_ref

from pyutilib.misc.indent_io import StreamIndenter

import pyomo.common
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.misc import tabular_writer, sorted_robust

logger = logging.getLogger('pyomo.core')

relocated_module_attribute(
    'ComponentUID', 'pyomo.core.base.componentuid.ComponentUID',
    version='5.7.2')

def _name_index_generator(idx):
    """
    Return a string representation of an index.
    """
    def _escape(val):
        if type(val) is tuple:
            ans = "(" + ','.join(_escape(_) for _ in val) + ")"
        else:
            # We need to quote set members (because people put things
            # like spaces - or worse commas - in their set names).  Our
            # plan is to put the strings in single quotes... but that
            # requires escaping any single quotes in the string... which
            # in turn requires escaping the escape character.
            ans = "%s" % (val,)
            if isinstance(val, six.string_types):
                ans = ans.replace("\\", "\\\\").replace("'", "\\'")
                if ',' in ans or "'" in ans:
                    ans = "'"+ans+"'"
        return ans
    if idx.__class__ is tuple:
        return "[" + ",".join(_escape(i) for i in idx) + "]"
    else:
        return "[" + _escape(idx) + "]"


def name(component, index=None, fully_qualified=False, relative_to=None):
    """
    Return a string representation of component for a specific
    index value.
    """
    base = component.getname(fully_qualified=fully_qualified, relative_to=relative_to)
    if index is None:
        return base
    else:
        if index not in component.index_set():
            raise KeyError( "Index %s is not valid for component %s"
                            % (index, component.name) )
        return base + _name_index_generator( index )


@deprecated(msg="The cname() function has been renamed to name()",
            version='5.6.9')
def cname(*args, **kwds):
    return name(*args, **kwds)


class CloneError(pyomo.common.errors.PyomoException):
    pass

class _ComponentBase(PyomoObject):
    """A base class for Component and ComponentData

    This class defines some fundamental methods and properties that are
    expected for all Component-like objects.  They are centralized here
    to avoid repeated code in the Component and ComponentData classes.
    """
    __slots__ = ()

    _PPRINT_INDENT = "    "

    def is_component_type(self):
        """Return True if this class is a Pyomo component"""
        return True

    def __deepcopy__(self, memo):
        # The problem we are addressing is when we want to clone a
        # sub-block in a model.  In that case, the block can have
        # references to both child components and to external
        # ComponentData (mostly through expressions pointing to Vars
        # and Params outside this block).  For everything stored beneath
        # this block, we want to clone the Component (and all
        # corresponding ComponentData objects).  But for everything
        # stored outside this Block, we want to do a simple shallow
        # copy.
        #
        # Nominally, expressions only point to ComponentData
        # derivatives.  However, with the development of Expression
        # Templates (and the corresponding _GetItemExpression object),
        # expressions can refer to container (non-Simple) components, so
        # we need to override __deepcopy__ for both Component and
        # ComponentData.

        #try:
        #    print("Component: %s" % (self.name,))
        #except:
        #    print("DANGLING ComponentData: %s on %s" % (
        #        type(self),self.parent_component()))

        # Note: there is an edge case when cloning a block: the initial
        # call to deepcopy (on the target block) has __block_scope__
        # defined, however, the parent block of self is either None, or
        # is (by definition) out of scope.  So we will check that
        # id(self) is not in __block_scope__: if it is, then this is the
        # top-level block and we need to do the normal deepcopy.
        if '__block_scope__' in memo and \
                id(self) not in memo['__block_scope__']:
            _known = memo['__block_scope__']
            _new = []
            tmp = self.parent_block()
            tmpId = id(tmp)
            # Note: normally we would need to check that tmp does not
            # end up being None.  However, since clone() inserts
            # id(None) into the __block_scope__ dictionary, we are safe
            while tmpId not in _known:
                _new.append(tmpId)
                tmp = tmp.parent_block()
                tmpId = id(tmp)

            # Remember whether all newly-encountered blocks are in or
            # out of scope (prevent duplicate work)
            for _id in _new:
                _known[_id] = _known[tmpId]

            if not _known[tmpId]:
                # component is out-of-scope.  shallow copy only
                ans = memo[id(self)] = self
                return ans

        #
        # There is a particularly subtle bug with 'uncopyable'
        # attributes: if the exception is thrown while copying a complex
        # data structure, we can be in a state where objects have been
        # created and assigned to the memo in the try block, but they
        # haven't had their state set yet.  When the exception moves us
        # into the except block, we need to effectively "undo" those
        # partially copied classes.  The only way is to restore the memo
        # to the state it was in before we started.  Right now, our
        # solution is to make a (shallow) copy of the memo before each
        # operation and restoring it in the case of exception.
        # Unfortunately that is a lot of usually unnecessary work.
        # Since *most* classes are copyable, we will avoid that
        # "paranoia" unless the naive clone generated an error - in
        # which case Block.clone() will switch over to the more
        # "paranoid" mode.
        #
        paranoid = memo.get('__paranoid__', None)

        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        # We can't do the "obvious", since this is a (partially)
        # slot-ized class and the __dict__ structure is
        # nonauthoritative:
        #
        # for key, val in self.__dict__.iteritems():
        #     object.__setattr__(ans, key, deepcopy(val, memo))
        #
        # Further, __slots__ is also nonauthoritative (this may be a
        # singleton component -- in which case it also has a __dict__).
        # Plus, as this may be a derived class with several layers of
        # slots.  So, we will resort to partially "pickling" the object,
        # deepcopying the state dict, and then restoring the copy into
        # the new instance.
        #
        # [JDS 7/7/14] I worry about the efficiency of using both
        # getstate/setstate *and* deepcopy, but we need deepcopy to
        # update the _parent refs appropriately, and since this is a
        # slot-ized class, we cannot overwrite the __deepcopy__
        # attribute to prevent infinite recursion.
        state = self.__getstate__()
        try:
            if paranoid:
                saved_memo = dict(memo)
            new_state = deepcopy(state, memo)
        except:
            if paranoid:
                # Note: memo is intentionally pass-by-reference.  We
                # need to clear and reset the object we were handed (and
                # not overwrite it)
                memo.clear()
                memo.update(saved_memo)
            elif paranoid is not None:
                raise PickleError()
            new_state = {}
            for k,v in iteritems(state):
                try:
                    if paranoid:
                        saved_memo = dict(memo)
                    new_state[k] = deepcopy(v, memo)
                except CloneError:
                    raise
                except:
                    if paranoid:
                        memo.clear()
                        memo.update(saved_memo)
                    elif paranoid is None:
                        logger.warning("""
                            Uncopyable field encountered when deep
                            copying outside the scope of Block.clone().
                            There is a distinct possibility that the new
                            copy is not complete.  To avoid this
                            situation, either use Block.clone() or set
                            'paranoid' mode by adding '__paranoid__' ==
                            True to the memo before calling
                            copy.deepcopy.""")
                    if self.model() is self:
                        what = 'Model'
                    else:
                        what = 'Component'
                    logger.error(
                        "Unable to clone Pyomo component attribute.\n"
                        "%s '%s' contains an uncopyable field '%s' (%s)"
                        % ( what, self.name, k, type(v) ))
                    # If this is an abstract model, then we are probably
                    # in the middle of create_instance, and the model
                    # that will eventually become the concrete model is
                    # missing initialization data.  This is an
                    # exceptional event worthy of a stronger (and more
                    # informative) error.
                    if not self.parent_component()._constructed:
                        raise CloneError(
                            "Uncopyable attribute (%s) encountered when "
                            "cloning component %s on an abstract block.  "
                            "The resulting instance is therefore "
                            "missing data from the original abstract model "
                            "and likely will not construct correctly.  "
                            "Consider changing how you initialize this "
                            "component or using a ConcreteModel."
                            % ( k, self.name ))
        ans.__setstate__(new_state)
        return ans

    @deprecated("""The cname() method has been renamed to getname().
    The preferred method of obtaining a component name is to use the
    .name property, which returns the fully qualified component name.
    The .local_name property will return the component name only within
    the context of the immediate parent container.""", version='5.0')
    def cname(self, *args, **kwds):
        return self.getname(*args, **kwds)

    def pprint(self, ostream=None, verbose=False, prefix=""):
        """Print component information

        Note that this method is generally only reachable through
        ComponentData objects in an IndexedComponent container.
        Components, including unindexed Component derivatives and both
        scalar and indexed IndexedComponent derivatives will see
        :py:meth:`Component.pprint()`
        """
        comp = self.parent_component()
        _attr, _data, _header, _fcn = comp._pprint()
        if isinstance(type(_data), six.string_types):
            # If the component _pprint only returned a pre-formatted
            # result, then we have no way to only emit the information
            # for this _data object.
            _name = comp.local_name
        else:
            # restrict output to only this data object
            _data = iter( ((self.index(), self),) )
            _name = "{Member of %s}" % (comp.local_name,)
        self._pprint_base_impl(
            ostream, verbose, prefix, _name, comp.doc,
            comp.is_constructed(), _attr, _data, _header, _fcn)

    @property
    def name(self):
        """Get the fully qualifed component name."""
        return self.getname(fully_qualified=True)

    # Adding a setter here to help users adapt to the new
    # setting. The .name attribute is now ._name. It should
    # never be assigned to by user code.
    @name.setter
    def name(self, val):
        raise ValueError(
            "The .name attribute is now a property method "
            "that returns the fully qualified component name. "
            "Assignment is not allowed.")

    @property
    def local_name(self):
        """Get the component name only within the context of
        the immediate parent container."""
        return self.getname(fully_qualified=False)

    @property
    def active(self):
        """Return the active attribute"""
        # Normal components cannot be deactivated
        return True

    @active.setter
    def active(self, value):
        """Set the active attribute to the given value"""
        raise AttributeError(
            "Setting the 'active' flag on a component that does not "
            "support deactivation is not allowed.")

    def _pprint_base_impl(self, ostream, verbose, prefix, _name, _doc,
                          _constructed, _attr, _data, _header, _fcn):
        if ostream is None:
            ostream = sys.stdout
        if prefix:
            ostream = StreamIndenter(ostream, prefix)

        # FIXME: HACK for backwards compatability with suppressing the
        # header for the top block
        if not _attr and self.parent_block() is None:
            _name = ''

        # We only indent everything if we printed the header
        if _attr or _name or _doc:
            ostream = StreamIndenter(ostream, self._PPRINT_INDENT)
            # The first line should be a hanging indent (i.e., not indented)
            ostream.newline = False

        if _name:
            ostream.write(_name+" : ")
        if _doc:
            ostream.write(_doc+'\n')
        if _attr:
            ostream.write(", ".join("%s=%s" % (k,v) for k,v in _attr))
        if _attr or _name or _doc:
            ostream.write("\n")

        if not _constructed:
            # HACK: for backwards compatability, Abstract blocks will
            # still print their assigned components.  Should we instead
            # always pprint unconstructed components (possibly
            # suppressing the table header if the table is empty)?
            if self.parent_block() is not None:
                ostream.write("Not constructed\n")
                return

        if type(_fcn) is tuple:
            _fcn, _fcn2 = _fcn
        else:
            _fcn2 = None

        if _header is not None:
            if _fcn2 is not None:
                _data_dict = dict(_data)
                _data = iteritems(_data_dict)
            tabular_writer( ostream, '', _data, _header, _fcn )
            if _fcn2 is not None:
                for _key in sorted_robust(_data_dict):
                    _fcn2(ostream, _key, _data_dict[_key])
        elif _fcn is not None:
            _data_dict = dict(_data)
            for _key in sorted_robust(_data_dict):
                _fcn(ostream, _key, _data_dict[_key])
        elif _data is not None:
            ostream.write(_data)


class Component(_ComponentBase):
    """
    This is the base class for all Pyomo modeling components.

    Constructor arguments:
        ctype           The class type for the derived subclass
        doc             A text string describing this component
        name            A name for this component

    Public class attributes:
        doc             A text string describing this component

    Private class attributes:
        _constructed    A boolean that is true if this component has been
                            constructed
        _parent         A weakref to the parent block that owns this component
        _ctype          The class type for the derived subclass
    """

    def __init__ (self, **kwds):
        #
        # Get arguments
        #
        self._ctype = kwds.pop('ctype', None)
        self.doc    = kwds.pop('doc', None)
        self._name  = kwds.pop('name', str(type(self).__name__))
        if kwds:
            raise ValueError(
                "Unexpected keyword options found while constructing '%s':\n\t%s"
                % ( type(self).__name__, ','.join(sorted(kwds.keys())) ))
        #
        # Verify that ctype has been specified.
        #
        if self._ctype is None:
            raise pyomo.common.DeveloperError(
                "Must specify a component type for class %s!"
                % ( type(self).__name__, ) )
        #
        self._constructed   = False
        self._parent        = None    # Must be a weakref

    def __getstate__(self):
        """
        This method must be defined to support pickling because this class
        owns weakrefs for '_parent'.
        """
        #
        # Nominally, __getstate__() should return:
        #
        # state = super(Class, self).__getstate__()
        # for i in Class.__dict__:
        #     state[i] = getattr(self,i)
        # return state
        #
        # However, in this case, the (nominal) parent class is 'object',
        # and object does not implement __getstate__.  So, we will check
        # to make sure that there is a base __getstate__() to call...
        #
        _base = super(Component,self)
        if hasattr(_base, '__getstate__'):
            state = _base.__getstate__()
            for key,val in iteritems(self.__dict__):
                if key not in state:
                    state[key] = val
        else:
            state = dict(self.__dict__)
        if self._parent is not None:
            state['_parent'] = self._parent()
        return state

    def __setstate__(self, state):
        """
        This method must be defined to support pickling because this class
        owns weakrefs for '_parent'.
        """
        if state['_parent'] is not None and \
                type(state['_parent']) is not weakref_ref:
            state['_parent'] = weakref_ref(state['_parent'])
        #
        # Note: our model for setstate is for derived classes to modify
        # the state dictionary as control passes up the inheritance
        # hierarchy (using super() calls).  All assignment of state ->
        # object attributes is handled at the last class before 'object'
        # (which may -- or may not (thanks to MRO) -- be here.
        #
        _base = super(Component,self)
        if hasattr(_base, '__setstate__'):
            _base.__setstate__(state)
        else:
            for key, val in iteritems(state):
                # Note: per the Python data model docs, we explicitly
                # set the attribute using object.__setattr__() instead
                # of setting self.__dict__[key] = val.
                object.__setattr__(self, key, val)

    @property
    def ctype(self):
        """Return the class type for this component"""
        return self._ctype

    @deprecated("Component.type() method has been replaced by the "
                ".ctype property.", version='5.7')
    def type(self):
        """Return the class type for this component"""
        return self.ctype

    def construct(self, data=None):                     #pragma:nocover
        """API definition for constructing components"""
        pass

    def is_constructed(self):                           #pragma:nocover
        """Return True if this class has been constructed"""
        return self._constructed

    def reconstruct(self, data=None):
        """Re-construct model expressions"""
        self._constructed = False
        self.construct(data=data)

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True

    def pprint(self, ostream=None, verbose=False, prefix=""):
        """Print component information"""
        self._pprint_base_impl(
            ostream, verbose, prefix, self.local_name, self.doc,
            self.is_constructed(), *self._pprint()
        )

    def display(self, ostream=None, verbose=False, prefix=""):
        self.pprint(ostream=ostream, prefix=prefix)

    def parent_component(self):
        """Returns the component associated with this object."""
        return self

    def parent_block(self):
        """Returns the parent of this object."""
        if self._parent is None:
            return None
        else:
            return self._parent()

    def model(self):
        """Returns the model associated with this object."""
        # This is a re-implementation of Component.parent_block(),
        # duplicated for effficiency to avoid the method call
        if self._parent is None:
            return None
        ans = self._parent()

        if ans is None:
            return None
        # Defer to the (simple) block's model() method to walk up the
        # hierarchy. This is because the top-level block can be a model,
        # but nothing else (e.g., calling model() on a Var not attached
        # to a model should return None, but calling model() on a Block
        # not attached to anything else should return the Block)
        return ans.model()

    def root_block(self):
        """Return self.model()"""
        return self.model()

    def __str__(self):
        """Return the component name"""
        return self.name

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """Return the component name"""
        if compute_values:
            try:
                return str(self())
            except:
                pass
        return self.name

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        """Returns the component name associated with this object.

        Parameters
        ----------
        fully_qualified: bool
            Generate full name from nested block names

        name_buffer: dict
            A dictionary that caches encountered names and indices.
            Providing a ``name_buffer`` can significantly speed up
            iterative name generation

        relative_to: Block
            Generate fully_qualified names reletive to the specified block.
        """
        if fully_qualified:
            pb = self.parent_block()
            if relative_to is None:
                relative_to = self.model()
            if pb is not None and pb is not relative_to:
                ans = pb.getname(fully_qualified, name_buffer, relative_to) \
                      + "." + self._name
            elif pb is None and relative_to != self.model():
                raise RuntimeError(
                    "The relative_to argument was specified but not found "
                    "in the block hierarchy: %s" % str(relative_to))
            else:
                ans = self._name
        else:
            ans = self._name
        if name_buffer is not None:
            name_buffer[id(self)] = ans
        return ans

    @property
    def name(self):
        """Get the fully qualifed component name."""
        return self.getname(fully_qualified=True)

    # Allow setting a componet's name if it is not owned by a parent
    # block (this supports, e.g., naming a model)
    @name.setter
    def name(self, val):
        if self.parent_block() is None:
            self._name = val
        else:
            raise ValueError(
                "The .name attribute is not settable when the component "
                "is assigned to a Block.\nTriggered by attempting to set "
                "component '%s' to name '%s'" % (self.name,val))

    def is_indexed(self):
        """Return true if this component is indexed"""
        return False

    def clear_suffix_value(self, suffix_or_name, expand=True):
        """Clear the suffix value for this component data"""
        if isinstance(suffix_or_name, six.string_types):
            import pyomo.core.base.suffix
            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
                if suffix_or_name == name_:
                    suffix_.clear_value(self, expand=expand)
                    break
        else:
            suffix_or_name.clear_value(self, expand=expand)

    def set_suffix_value(self, suffix_or_name, value, expand=True):
        """Set the suffix value for this component data"""
        if isinstance(suffix_or_name, six.string_types):
            import pyomo.core.base.suffix
            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
                if suffix_or_name == name_:
                    suffix_.set_value(self, value, expand=expand)
                    break
        else:
            suffix_or_name.set_value(self, value, expand=expand)

    def get_suffix_value(self, suffix_or_name, default=None):
        """Get the suffix value for this component data"""
        if isinstance(suffix_or_name, six.string_types):
            import pyomo.core.base.suffix
            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
                if suffix_or_name == name_:
                    return suffix_.get(self, default)
        else:
            return suffix_or_name.get(self, default)


class ActiveComponent(Component):
    """A Component that makes semantic sense to activate or deactivate
    in a model.

    Private class attributes:
        _active         A boolean that is true if this component will be
                            used in model operations
    """

    def __init__(self, **kwds):
        self._active = True
        super(ActiveComponent, self).__init__(**kwds)

    @property
    def active(self):
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active attribute to the given value"""
        raise AttributeError(
            "Assignment not allowed. Use the (de)activate methods." )

    def activate(self):
        """Set the active attribute to True"""
        self._active=True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active=False


class ComponentData(_ComponentBase):
    """
    This is the base class for the component data used
    in Pyomo modeling components.  Subclasses of ComponentData are
    used in indexed components, and this class assumes that indexed
    components are subclasses of IndexedComponent.  Note that
    ComponentData instances do not store their index.  This makes
    some operations significantly more expensive, but these are (a)
    associated with I/O generation and (b) this cost can be managed
    with caches.

    Constructor arguments:
        owner           The component that owns this data object

    Private class attributes:
        _component      A weakref to the component that owns this data object
        """

    __pickle_slots__ = ('_component',)
    __slots__ = __pickle_slots__ + ('__weakref__',)

    def __init__(self, component):
        #
        # ComponentData objects are typically *private* objects for
        # indexed / sparse indexed components.  As such, the (derived)
        # class needs to make sure that the owning component is *always*
        # passed as the owner (and that owner is never None).  Not validating
        # this assumption is significantly faster.
        #
        self._component = weakref_ref(component)

    def __getstate__(self):
        """Prepare a picklable state of this instance for pickling.

        Nominally, __getstate__() should return:

            state = super(Class, self).__getstate__()
            for i in Class.__slots__:
                state[i] = getattr(self,i)
            return state

        However, in this case, the (nominal) parent class is 'object',
        and object does not implement __getstate__.  So, we will check
        to make sure that there is a base __getstate__() to call...
        You might think that there is nothing to check, but multiple
        inheritance could mean that another class got stuck between
        this class and "object" in the MRO.

        This method must be defined to support pickling because this
        class owns weakrefs for '_component', which must be either
        removed or converted to hard references prior to pickling.

        Further, since there is only a single slot, and that slot
        (_component) requires special processing, we will just deal with
        it explicitly.  As _component is a weakref (not pickable), we
        need to resolve it to a concrete object.
        """
        _base = super(ComponentData,self)
        if hasattr(_base, '__getstate__'):
            state = _base.__getstate__()
        else:
            state = {}
        #
        if self._component is None:
            state['_component'] = None
        else:
            state['_component'] = self._component()
        return state

    def __setstate__(self, state):
        """Restore a pickled state into this instance

        Note: our model for setstate is for derived classes to modify
        the state dictionary as control passes up the inheritance
        hierarchy (using super() calls).  All assignment of state ->
        object attributes is handled at the last class before 'object'
        (which may -- or may not (thanks to MRO) -- be here.

        This method must be defined to support unpickling because this
        class owns weakrefs for '_component', which must be restored
        from the hard references used in the piclke.
        """
        #
        # FIXME: We shouldn't have to check for weakref.ref here, but if
        # we don't the model cloning appears to fail (in the Benders
        # example)
        #
        if state['_component'] is not None and \
                type(state['_component']) is not weakref_ref:
            state['_component'] = weakref_ref(state['_component'])
        #
        # Note: our model for setstate is for derived classes to modify
        # the state dictionary as control passes up the inheritance
        # hierarchy (using super() calls).  All assignment of state ->
        # object attributes is handled at the last class before 'object'
        # (which may -- or may not (thanks to MRO) -- be here.
        #
        _base = super(ComponentData,self)
        if hasattr(_base, '__setstate__'):
            _base.__setstate__(state)
        else:
            for key, val in iteritems(state):
                # Note: per the Python data model docs, we explicitly
                # set the attribute using object.__setattr__() instead
                # of setting self.__dict__[key] = val.
                object.__setattr__(self, key, val)

    @property
    def ctype(self):
        """Return the class type for this component"""
        _parent = self.parent_component()
        if _parent is None:
            return None
        return _parent._ctype

    @deprecated("Component.type() method has been replaced by the "
                ".ctype property.", version='5.7')
    def type(self):
        """Return the class type for this component"""
        return self.ctype

    def parent_component(self):
        """Returns the component associated with this object."""
        if self._component is None:
            return None
        return self._component()

    def parent_block(self):
        """Return the parent of the component that owns this data. """
        # This is a re-implementation of parent_component(), duplicated
        # for effficiency to avoid the method call
        if self._component is None:
            return None
        comp = self._component()

        # This is a re-implementation of Component.parent_block(),
        # duplicated for effficiency to avoid the method call
        if comp._parent is None:
            return None
        return comp._parent()

    def model(self):
        """Return the model of the component that owns this data. """
        ans = self.parent_block()
        if ans is None:
            return None
        # Defer to the (simple) block's model() method to walk up the
        # hierarchy. This is because the top-level block can be a model,
        # but nothing else (e.g., calling model() on a Var not attached
        # to a model should return None, but calling model() on a Block
        # not attached to anything else should return the Block)
        return ans.model()

    def index(self):
        """
        Returns the index of this ComponentData instance relative
        to the parent component index set. None is returned if
        this instance does not have a parent component, or if
        - for some unknown reason - this instance does not belong
        to the parent component's index set. This method is not
        intended to be a fast method;  it should be used rarely,
        primarily in cases of label formulation.
        """
        self_component = self.parent_component()
        if self_component is None:
            return None
        for idx, component_data in self_component.iteritems():
            if component_data is self:
                return idx
        return None

    def __str__(self):
        """Return a string with the component name and index"""
        return self.name

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """
        Return a string representation of this component,
        applying the labeler if passed one.
        """
        if compute_values:
            try:
                return str(self())
            except:
                pass
        if smap:
            return smap.getSymbol(self, labeler)
        if labeler is not None:
            return labeler(self)
        else:
            return self.__str__()

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        """Return a string with the component name and index"""
        #
        # Using the buffer, which is a dictionary:  id -> string
        #
        if name_buffer is not None and id(self) in name_buffer:
            # Return the name if it is in the buffer
            return name_buffer[id(self)]

        c = self.parent_component()
        if c is self:
            #
            # This is a scalar component, so call the
            # Component.getname() method
            #
            return super(ComponentData, self).getname(
                fully_qualified, name_buffer, relative_to)
        elif c is not None:
            #
            # Get the name of the parent component
            #
            base = c.getname(fully_qualified, name_buffer, relative_to)
        else:
            #
            # Defensive: this is a ComponentData without a valid
            # parent_component.  As this usually occurs when handling
            # exceptions during model construction, we need to ensure
            # that this method doesn't itself raise another exception.
            #
            return '[Unattached %s]' % (type(self).__name__,)

        if name_buffer is not None:
            # Iterate through the dictionary and generate all names in
            # the buffer
            for idx, obj in iteritems(c):
                name_buffer[id(obj)] = base + _name_index_generator(idx)
            if id(self) in name_buffer:
                # Return the name if it is in the buffer
                return name_buffer[id(self)]
        else:
            #
            # No buffer, so we iterate through the component _data
            # dictionary until we find this object.  This can be much
            # more expensive than if a buffer is provided.
            #
            for idx, obj in iteritems(c):
                if obj is self:
                    return base + _name_index_generator(idx)
        #
        raise RuntimeError("Fatal error: cannot find the component data in "
                           "the owning component's _data dictionary.")

    def is_indexed(self):
        """Return true if this component is indexed"""
        return False

    def clear_suffix_value(self, suffix_or_name, expand=True):
        """Set the suffix value for this component data"""
        if isinstance(suffix_or_name, six.string_types):
            import pyomo.core.base.suffix
            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
                if suffix_or_name == name_:
                    suffix_.clear_value(self, expand=expand)
                    break
        else:
            suffix_or_name.clear_value(self, expand=expand)

    def set_suffix_value(self, suffix_or_name, value, expand=True):
        """Set the suffix value for this component data"""
        if isinstance(suffix_or_name, six.string_types):
            import pyomo.core.base.suffix
            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
                if suffix_or_name == name_:
                    suffix_.set_value(self, value, expand=expand)
                    break
        else:
            suffix_or_name.set_value(self, value, expand=expand)

    def get_suffix_value(self, suffix_or_name, default=None):
        """Get the suffix value for this component data"""
        if isinstance(suffix_or_name, six.string_types):
            import pyomo.core.base.suffix
            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
                if suffix_or_name == name_:
                    return suffix_.get(self, default)
        else:
            return suffix_or_name.get(self, default)


class ActiveComponentData(ComponentData):
    """
    This is the base class for the component data used
    in Pyomo modeling components that can be activated and
    deactivated.

    It's possible to end up in a state where the parent Component
    has _active=True but all ComponentData have _active=False. This
    seems like a reasonable state, though we cannot easily detect
    this situation.  The important thing to avoid is the situation
    where one or more ComponentData are active, but the parent
    Component claims active=False. This class structure is designed
    to prevent this situation.

    Constructor arguments:
        owner           The component that owns this data object

    Private class attributes:
        _component      A weakref to the component that owns this data object
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ( '_active', )

    def __init__(self, component):
        super(ActiveComponentData, self).__init__(component)
        self._active = True

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(ActiveComponentData, self).__getstate__()
        for i in ActiveComponentData.__slots__:
            result[i] = getattr(self, i)
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    @property
    def active(self):
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active attribute to a specified value."""
        raise AttributeError(
            "Assignment not allowed. Use the (de)activate method" )

    def activate(self):
        """Set the active attribute to True"""
        self._active = self.parent_component()._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False

