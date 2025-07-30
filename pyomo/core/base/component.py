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

import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref

import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
    RenamedClass,
    deprecated,
    deprecation_warning,
    relocated_module_attribute,
)
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import PartialInitializer

logger = logging.getLogger('pyomo.core')

relocated_module_attribute(
    'ComponentUID', 'pyomo.core.base.componentuid.ComponentUID', version='5.7.2'
)

_ref_types = {type(None), weakref_ref}


class ModelComponentFactoryClass(Factory):
    def register(self, doc=None):
        def fn(cls):
            return super(ModelComponentFactoryClass, self).register(cls.__name__, doc)(
                cls
            )

        return fn


ModelComponentFactory = ModelComponentFactoryClass('model component')


def name(component, index=NOTSET, fully_qualified=False, relative_to=None):
    """
    Return a string representation of component for a specific
    index value.
    """
    base = component.getname(fully_qualified=fully_qualified, relative_to=relative_to)
    if index is NOTSET:
        return base
    else:
        if index not in component.index_set():
            raise KeyError(
                "Index %s is not valid for component %s" % (index, component.name)
            )
        return base + index_repr(index)


@deprecated(msg="The cname() function has been renamed to name()", version='5.6.9')
def cname(*args, **kwds):
    return name(*args, **kwds)


class CloneError(pyomo.common.errors.PyomoException):
    pass


class ComponentBase(PyomoObject):
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
        # ComponentData (so we put it here on ComponentBase).
        #
        if '__block_scope__' in memo:
            _scope = memo['__block_scope__']
            _new = None
            tmp = self.parent_block()
            # "Floating" components should be in scope by default (we
            # will handle 'global' components like GlobalSets in the
            # components).  This ensures that things like set operators
            # on Abstract set objects are correctly cloned.  For
            # example, consider an abstract indexed model component
            # whose domain is specified by a Set expression:
            #
            #   def x_init(m,i):
            #       if i == 2:
            #           return Set.Skip
            #       else:
            #           return []
            #   m.x = Set( [1,2],
            #              domain={1: m.A*m.B, 2: m.A*m.A},
            #              initialize=x_init )
            #
            # We do not want to automatically add all the Set operators
            # to the model at declaration time, as m.x[2] is never
            # actually created.  Plus, doing so would require complex
            # parsing of the initializers.  BUT, we need to ensure that
            # the operators are deepcopied, otherwise when the model is
            # cloned before construction the operators will still refer
            # to the sets on the original abstract model (in particular,
            # the Set x will have an unknown dimen).
            #
            # The solution is to automatically clone all floating
            # components, except for Models (i.e., top-level BlockData
            # have no parent and technically "float")
            _in_scope = tmp is None and self is not self.model()
            #
            # Note: normally we would need to check that tmp does not
            # end up being None.  However, since clone() inserts
            # id(None) into the __block_scope__ dictionary, we are safe
            while id(tmp) not in _scope:
                _new = (_new, id(tmp))
                tmp = tmp.parent_block()
            _in_scope |= _scope[id(tmp)]

            # Remember whether all newly-encountered blocks are in or
            # out of scope (prevent duplicate work)
            while _new is not None:
                _new, _id = _new
                _scope[_id] = _in_scope

            # Note: there is an edge case when cloning a block: the
            # initial call to deepcopy (on the target block) has
            # __block_scope__ defined, however, the parent block of self
            # is either None, or is (by definition) out of scope.  So we
            # will check that id(self) is not in __block_scope__: if it
            # is, then this is the top-level block and we need to do the
            # normal deepcopy.  We defer this check until now for
            # efficiency reasons because we expect that (for sane models)
            # the bulk of the components we will encounter will be *in*
            # scope.
            if not _in_scope and id(self) not in _scope:
                # component is out-of-scope.  shallow copy only
                memo[id(self)] = self
                return self
        #
        # deepcopy() is an inherently recursive operation.  This can
        # cause problems for highly interconnected Pyomo models (for
        # example, a time linked model where each time block has a
        # linking constraint [in the time block] to the next / previous
        # block).  This would effectively put the entire time horizon on
        # the stack.  To avoid this, we will leverage the useful
        # knowledge that all component references point to other
        # components / component datas, and NOT to attributes on the
        # components/datas.  So, if we can first go through and stub in
        # all the objects that we will need to populate, and then go
        # through and deepcopy them, we can unroll the vast
        # majority of the recursion.
        #
        component_list = []
        self._create_objects_for_deepcopy(memo, component_list)
        #
        # Note that self is now the first element of component_list
        #
        # Now that we have created (but not populated) all the
        # components that we expect to need in the memo, we can go
        # through and populate all the components.
        #
        # The component_list is roughly in declaration order.  This
        # means that it should be relatively safe to clone the contents
        # in the same order.
        #
        for comp, new in component_list:
            comp.__deepcopy_state__(memo, new)
        return memo[id(self)]

    def _create_objects_for_deepcopy(self, memo, component_list):
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append((self, _new))
        return _ans

    def __deepcopy_field__(self, value, memo, slot_name):
        saved_memo = len(memo)
        try:
            return fast_deepcopy(value, memo)
        except CloneError:
            raise
        except:
            # remove entries added to the memo
            for _ in range(len(memo) - saved_memo):
                memo.popitem()
            # warn the user
            if '__block_scope__' not in memo:
                logger.warning(
                    "Uncopyable field encountered when deep "
                    "copying Pyomo components outside the scope of "
                    "Block.clone().  There is a distinct possibility "
                    "that the new copy is not complete.  To avoid "
                    "this situation, please use Block.clone()"
                )
            if self.model() is self:
                what = 'Model'
            else:
                what = 'Component'
            logger.error(
                "Unable to clone Pyomo component attribute.\n"
                "%s '%s' contains an uncopyable field '%s' (%s).  "
                "Setting field to `None` on new object"
                % (what, self.name, slot_name, type(value))
            )
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
                    "component or using a ConcreteModel." % (slot_name, self.name)
                )
        # Drop the offending field value.  The user has been warned.
        return None

    @deprecated(
        """The cname() method has been renamed to getname().
    The preferred method of obtaining a component name is to use the
    .name property, which returns the fully qualified component name.
    The .local_name property will return the component name only within
    the context of the immediate parent container.""",
        version='5.0',
    )
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
        if isinstance(type(_data), str):
            # If the component _pprint only returned a pre-formatted
            # result, then we have no way to only emit the information
            # for this _data object.
            _name = comp.local_name
        else:
            # restrict output to only this data object
            _data = iter(((self.index(), self),))
            _name = "{Member of %s}" % (comp.local_name,)
        self._pprint_base_impl(
            ostream,
            verbose,
            prefix,
            _name,
            comp.doc,
            comp.is_constructed(),
            _attr,
            _data,
            _header,
            _fcn,
        )

    @property
    def name(self):
        """Get the fully qualified component name."""
        return self.getname(fully_qualified=True)

    # Adding a setter here to help users adapt to the new
    # setting. The .name attribute is now ._name. It should
    # never be assigned to by user code.
    @name.setter
    def name(self, val):
        raise ValueError(
            "The .name attribute is now a property method "
            "that returns the fully qualified component name. "
            "Assignment is not allowed."
        )

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
            "support deactivation is not allowed."
        )

    def _pprint_base_impl(
        self,
        ostream,
        verbose,
        prefix,
        _name,
        _doc,
        _constructed,
        _attr,
        _data,
        _header,
        _fcn,
    ):
        if ostream is None:
            ostream = sys.stdout
        if prefix:
            ostream = StreamIndenter(ostream, prefix)

        # FIXME: HACK for backwards compatibility with suppressing the
        # header for the top block
        if not _attr and self.parent_block() is None:
            _name = ''

        # We only indent everything if we printed the header
        if _attr or _name or _doc:
            ostream = StreamIndenter(ostream, self._PPRINT_INDENT)
            # The first line should be a hanging indent (i.e., not indented)
            ostream.newline = False

        if self.is_reference():
            _attr = list(_attr) if _attr else []
            _attr.append(('ReferenceTo', self.referent))

        if _name:
            ostream.write(_name + " : ")
        if _doc:
            ostream.write(_doc + '\n')
        if _attr:
            ostream.write(", ".join("%s=%s" % (k, v) for k, v in _attr))
        if _attr or _name or _doc:
            ostream.write("\n")

        if not _constructed:
            # HACK: for backwards compatibility, Abstract blocks will
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
                _data = _data_dict.items()
            tabular_writer(ostream, '', _data, _header, _fcn)
            if _fcn2 is not None:
                for _key in sorted_robust(_data_dict):
                    _fcn2(ostream, _key, _data_dict[_key])
        elif _fcn is not None:
            _data_dict = dict(_data)
            for _key in sorted_robust(_data_dict):
                _fcn(ostream, _key, _data_dict[_key])
        elif _data is not None:
            ostream.write(_data)


class _ComponentBase(metaclass=RenamedClass):
    __renamed__new_class__ = ComponentBase
    __renamed__version__ = '6.7.2'


class Component(ComponentBase):
    """
    This is the base class for all Pyomo modeling components.

    Parameters
    ----------
    ctype : type
        The class type for the derived subclass

    doc : str
        A text string describing this component

    name : str
        A name for this component

    Attributes
    ----------
    doc : str
        A text string describing this component

    """

    __autoslot_mappers__ = {'_parent': AutoSlots.weakref_mapper}

    def __init__(self, **kwds):
        #
        # Get arguments
        #
        self._ctype = kwds.pop('ctype', None)
        self.doc = kwds.pop('doc', None)
        self._name = kwds.pop('name', None)
        if kwds:
            # If there are leftover keywords, and the component has a
            # rule, pass those keywords on to the rule
            if getattr(self, '_rule', None) is not None:
                self._rule = PartialInitializer(self._rule, **kwds)
            else:
                raise ValueError(
                    "Unexpected keyword options found while constructing '%s':\n\t%s"
                    % (type(self).__name__, ','.join(sorted(kwds.keys())))
                )
        #
        # Verify that ctype has been specified.
        #
        if self._ctype is None:
            raise DeveloperError(
                "Must specify a component type for class %s!" % (type(self).__name__,)
            )
        #
        self._constructed = False
        self._parent = None  # Must be a weakref

    @property
    def ctype(self):
        """Return the class type for this component"""
        return self._ctype

    @deprecated(
        "Component.type() method has been replaced by the .ctype property.",
        version='5.7',
    )
    def type(self):
        """Return the class type for this component"""
        return self.ctype

    def construct(self, data=None):  # pragma:nocover
        """API definition for constructing components"""
        pass

    def is_constructed(self):  # pragma:nocover
        """Return True if this class has been constructed"""
        return self._constructed

    def reconstruct(self, data=None):
        """REMOVED: reconstruct() was removed in Pyomo 6.0.

        Re-constructing model components was fragile and did not
        correctly update instances of the component used in other
        components or contexts (this was particularly problemmatic for
        Var, Param, and Set).  Users who wish to reproduce the old
        behavior of reconstruct(), are comfortable manipulating
        non-public interfaces, and who take the time to verify that the
        correct thing happens to their model can approximate the old
        behavior of reconstruct with:

            component.clear()
            component._constructed = False
            component.construct()

        """
        raise AttributeError(self.reconstruct.__doc__)

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True

    def pprint(self, ostream=None, verbose=False, prefix=""):
        """Print component information"""
        self._pprint_base_impl(
            ostream,
            verbose,
            prefix,
            self.local_name,
            self.doc,
            self.is_constructed(),
            *self._pprint(),
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

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        """Returns the component name associated with this object.

        Parameters
        ----------
        fully_qualified: bool
            Generate full name from nested block names

        relative_to: Block
            Generate fully_qualified names relative to the specified block.
        """
        local_name = self._name
        if local_name is None:
            local_name = type(self).__name__
        if fully_qualified:
            pb = self.parent_block()
            if relative_to is None:
                relative_to = self.model()
            if pb is not None and pb is not relative_to:
                ans = (
                    pb.getname(fully_qualified, name_buffer, relative_to)
                    + "."
                    + name_repr(local_name)
                )
            elif pb is None and relative_to != self.model():
                raise RuntimeError(
                    "The relative_to argument was specified but not found "
                    "in the block hierarchy: %s" % str(relative_to)
                )
            else:
                ans = name_repr(local_name)
        else:
            # Note: we want "getattr(x.parent_block(), x.local_name) == x"
            # so we do not want to call _safe_name_str, as that could
            # add quotes or otherwise escape the string.
            ans = local_name
        if name_buffer is not None:
            deprecation_warning(
                "The 'name_buffer' argument to getname is deprecated. "
                "The functionality is no longer necessary since getting names "
                "is no longer a quadratic operation. Additionally, note that "
                "use of this argument poses risks if the buffer contains "
                "names relative to different Blocks in the model hierarchy or "
                "a mixture of local and fully_qualified names.",
                version='6.4.1',
            )
            name_buffer[id(self)] = ans
        return ans

    @property
    def name(self):
        """Get the fully qualified component name."""
        return self.getname(fully_qualified=True)

    # Allow setting a component's name if it is not owned by a parent
    # block (this supports, e.g., naming a model)
    @name.setter
    def name(self, val):
        if self.parent_block() is None:
            self._name = val
        else:
            raise ValueError(
                "The .name attribute is not settable when the component "
                "is assigned to a Block.\nTriggered by attempting to set "
                "component '%s' to name '%s'" % (self.name, val)
            )

    def is_indexed(self):
        """Return true if this component is indexed"""
        return False

    def clear_suffix_value(self, suffix_or_name, expand=True):
        """Clear the suffix value for this component data"""
        if isinstance(suffix_or_name, str):
            import pyomo.core.base.suffix

            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(
                self.model()
            ):
                if suffix_or_name == name_:
                    suffix_.clear_value(self, expand=expand)
                    break
        else:
            suffix_or_name.clear_value(self, expand=expand)

    def set_suffix_value(self, suffix_or_name, value, expand=True):
        """Set the suffix value for this component data"""
        if isinstance(suffix_or_name, str):
            import pyomo.core.base.suffix

            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(
                self.model()
            ):
                if suffix_or_name == name_:
                    suffix_.set_value(self, value, expand=expand)
                    break
        else:
            suffix_or_name.set_value(self, value, expand=expand)

    def get_suffix_value(self, suffix_or_name, default=None):
        """Get the suffix value for this component data"""
        if isinstance(suffix_or_name, str):
            import pyomo.core.base.suffix

            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(
                self.model()
            ):
                if suffix_or_name == name_:
                    return suffix_.get(self, default)
        else:
            return suffix_or_name.get(self, default)

    def _pop_from_kwargs(self, name, kwargs, namelist, notset=None):
        args = [
            arg
            for arg in (kwargs.pop(name, notset) for name in namelist)
            if arg is not notset
        ]
        if len(args) == 1:
            return args[0]
        elif not args:
            return notset
        else:
            argnames = "%s%s '%s='" % (
                ', '.join("'%s='" % _ for _ in namelist[:-1]),
                ',' if len(namelist) > 2 else '',
                namelist[-1],
            )
            raise ValueError(
                "Duplicate initialization: %s() only accepts one of %s"
                % (name, argnames)
            )


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
        raise AttributeError("Assignment not allowed. Use the (de)activate methods.")

    def activate(self):
        """Set the active attribute to True"""
        self._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False


class ComponentData(ComponentBase):
    """
    This is the base class for the component data used
    in Pyomo modeling components.  Subclasses of ComponentData are
    used in indexed components, and this class assumes that indexed
    components are subclasses of IndexedComponent.

    Constructor arguments:
        owner           The component that owns this data object

    Private class attributes:
        _component      A weakref to the component that owns this data object
        _index          The index of this data object
    """

    __slots__ = ('_component', '_index', '__weakref__')
    __autoslot_mappers__ = {'_component': AutoSlots.weakref_mapper}

    # NOTE: This constructor is in-lined in the constructors for the following
    # classes: BooleanVarData, ConnectorData, ConstraintData,
    # ExpressionData, LogicalConstraintData,
    # LogicalConstraintData, ObjectiveData,
    # ParamData,VarData, BooleanVarData, DisjunctionData,
    # ArcData, PortData, _LinearConstraintData, and
    # _LinearMatrixConstraintData. Changes made here need to be made in those
    # constructors as well!
    def __init__(self, component):
        #
        # ComponentData objects are typically *private* objects for
        # indexed / sparse indexed components.  As such, the (derived)
        # class needs to make sure that the owning component is *always*
        # passed as the owner (and that owner is never None).  Not validating
        # this assumption is significantly faster.
        #
        self._component = weakref_ref(component)
        self._index = NOTSET

    @property
    def ctype(self):
        """Return the class type for this component"""
        _parent = self.parent_component()
        if _parent is None:
            return None
        return _parent._ctype

    @deprecated(
        "Component.type() method has been replaced by the .ctype property.",
        version='5.7',
    )
    def type(self):
        """Return the class type for this component"""
        return self.ctype

    def parent_component(self):
        """Returns the component associated with this object."""
        if self._component is None:
            return None
        return self._component()

    def parent_block(self):
        """Return the parent of the component that owns this data."""
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
        """Return the model of the component that owns this data."""
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
        to the parent component's index set.
        """
        try:
            if self._component()[self._index] is self:
                return self._index
        except:
            pass
        if self._index is NOTSET:
            return self._index
        parent = self.parent_component()
        if parent is None:
            return self._index
        # This error message is a bit goofy, but we can't call self.name
        # here--it's an infinite loop!
        raise DeveloperError(
            "The '_data' dictionary and '_index' attribute are out of "
            "sync for indexed %s '%s': The %s entry in the '_data' "
            "dictionary does not map back to this component data object."
            % (parent.ctype.__name__, parent.name, self._index)
        )

    def __str__(self):
        """Return a string with the component name and index"""
        return self.name

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        """Return a string with the component name and index"""
        # NOTE: There are bugs with name buffers if a user always gives the same
        # dictionary but switches from fully-qualified to local names or changes
        # the component the name is relative to. We will simply deprecate the
        # buffer in a future PR, but for now we will leave this method so it
        # behaves the same when a buffer is given, and, in the absence of a
        # buffer it will construct the name using the index (woohoo!)

        #
        # Using the buffer, which is a dictionary:  id -> string
        #
        if name_buffer is not None:
            deprecation_warning(
                "The 'name_buffer' argument to getname is deprecated. "
                "The functionality is no longer necessary since getting names "
                "is no longer a quadratic operation. Additionally, note that "
                "use of this argument poses risks if the buffer contains "
                "names relative to different Blocks in the model hierarchy or "
                "a mixture of local and fully_qualified names.",
                version='6.4.1',
            )
            if id(self) in name_buffer:
                # Return the name if it is in the buffer
                return name_buffer[id(self)]

        c = self.parent_component()
        if c is self:
            #
            # This is a scalar component, so call the
            # Component.getname() method
            #
            return super(ComponentData, self).getname(
                fully_qualified, name_buffer, relative_to
            )
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
            for idx, obj in c.items():
                name_buffer[id(obj)] = base + index_repr(idx)
            if id(self) in name_buffer:
                # Return the name if it is in the buffer
                return name_buffer[id(self)]
        else:
            #
            # No buffer, we can do what we are going to do all the time after we
            # deprecate the buffer.
            #
            return base + index_repr(self.index())
        #
        raise RuntimeError(
            "Fatal error: cannot find the component data in "
            "the owning component's _data dictionary."
        )

    def is_indexed(self):
        """Return true if this component is indexed"""
        return False

    def clear_suffix_value(self, suffix_or_name, expand=True):
        """Set the suffix value for this component data"""
        if isinstance(suffix_or_name, str):
            import pyomo.core.base.suffix

            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(
                self.model()
            ):
                if suffix_or_name == name_:
                    suffix_.clear_value(self, expand=expand)
                    break
        else:
            suffix_or_name.clear_value(self, expand=expand)

    def set_suffix_value(self, suffix_or_name, value, expand=True):
        """Set the suffix value for this component data"""
        if isinstance(suffix_or_name, str):
            import pyomo.core.base.suffix

            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(
                self.model()
            ):
                if suffix_or_name == name_:
                    suffix_.set_value(self, value, expand=expand)
                    break
        else:
            suffix_or_name.set_value(self, value, expand=expand)

    def get_suffix_value(self, suffix_or_name, default=None):
        """Get the suffix value for this component data"""
        if isinstance(suffix_or_name, str):
            import pyomo.core.base.suffix

            for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(
                self.model()
            ):
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
        _index          The index of this data object
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ('_active',)

    def __init__(self, component):
        super(ActiveComponentData, self).__init__(component)
        self._active = True

    @property
    def active(self):
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active attribute to a specified value."""
        raise AttributeError("Assignment not allowed. Use the (de)activate method")

    def activate(self):
        """Set the active attribute to True"""
        self._active = self.parent_component()._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False
