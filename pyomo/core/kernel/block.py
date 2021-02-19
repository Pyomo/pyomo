#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import logging
import math

from pyomo.common.collections import OrderedDict
if sys.version_info[:2] >= (3,7):
    # dict became ordered in CPython 3.6 and added to the standard in 3.7
    _ordered_dict_ = dict
else:
    _ordered_dict_ = OrderedDict

from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.base import \
    (_no_ctype,
     _convert_ctype)
from pyomo.core.kernel.heterogeneous_container import \
    IHeterogeneousContainer
from pyomo.core.kernel.container_utils import \
    define_simple_containers

import six

logger = logging.getLogger('pyomo.core')

class IBlock(IHeterogeneousContainer):
    """A generalized container that can store objects of
    any category type as attributes.
    """
    __slots__ = ()
    _child_storage_delimiter_string = "."
    _child_storage_entry_string = "%s"

    #
    # Define the IHeterogeneousContainer abstract methods
    #

    #def child_ctypes(self, *args, **kwds):
    # ... not defined here

    #
    # Define the ICategorizedObjectContainer abstract methods
    #

    def child(self, key):
        """Get the child object associated with a given
        storage key for this container.

        Raises:
            KeyError: if the argument is not a storage key
                for any children of this container
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(str(key))

    #def children(self, *args, **kwds):
    # ... not defined here

class block(IBlock):
    """A generalized container for defining hierarchical
    models by adding modeling components as attributes.

    Examples:
        >>> import pyomo.kernel as pmo
        >>> model = pmo.block()
        >>> model.x = pmo.variable()
        >>> model.c = pmo.constraint(model.x >= 1)
        >>> model.o = pmo.objective(model.x)
    """

    _ctype = IBlock
    # determines when large block storage is activated
    _lbs_count = 4
    _block_reserved_words = set()

    @staticmethod
    def _refresh_block_reserved_words():
        block._block_reserved_words = set(dir(block()))
        # this is protected because it is
        # defined as a property with a setter
        # that explicitly disallows this
        block._block_reserved_words.remove('active')

    def __init__(self):
        # bypass __setatrr__ for this class
        d = self.__dict__
        d['_parent'] = None
        d['_storage_key'] = None
        d['_active'] = True
        d['_block__byctype'] = None
        d['_block__order'] = _ordered_dict_()

    def _activate_large_storage_mode(self):
        if self.__byctype.__class__ is not _ordered_dict_:
            self_byctype = \
                self.__dict__['_block__byctype'] = _ordered_dict_()
            for key, obj in self.__order.items():
                ctype = obj.ctype
                if ctype not in self_byctype:
                    self_byctype[ctype] = _ordered_dict_()
                self_byctype[ctype][key] = obj

    #
    # Define the IHeterogeneousContainer abstract methods
    #

    def child_ctypes(self):
        """Returns the set of child object category types
        stored in this container."""
        self_byctype = self.__byctype
        if self_byctype is None:
            # empty
            return ()
        elif self_byctype.__class__ is int:
            # small block storage
            # (self_byctype is a union of hash bytes)
            ctypes_set = set()
            ctypes = []
            for child in self.__order.values():
                child_ctype = child.ctype
                if child_ctype not in ctypes_set:
                    ctypes_set.add(child_ctype)
                    ctypes.append(child_ctype)
            return tuple(ctypes)
        elif self_byctype.__class__ is _ordered_dict_:
            # large-block storage
            return tuple(self_byctype)
        else:
            # storing a single ctype
            return (self_byctype,)

    #
    # Define the ICategorizedObjectContainer abstract methods
    #

    def children(self, ctype=_no_ctype):
        """Iterate over the children of this block.

        Args:
            ctype: Indicates the category of children to
                include. The default value indicates that
                all categories should be included.

        Returns:
            iterator of child objects
        """
        self_byctype = self.__byctype
        if self_byctype is None:
            # empty
            return

        # convert AML types into Kernel types (hack for the
        # solver interfaces)
        ctype = _convert_ctype.get(ctype, ctype)

        if ctype is _no_ctype:
            for child in self.__order.values():
                yield child
        elif self_byctype.__class__ is _ordered_dict_:
            # large-block storage
            if ctype in self_byctype:
                for child in self_byctype[ctype].values():
                    yield child
        elif self_byctype.__class__ is int:
            # small-block storage
            # (self_byctype is a union of hash bytes)
            h_ = hash(ctype)
            if (self_byctype & h_) == h_:
                for child in self.__order.values():
                    if child.ctype is ctype:
                        yield child
        elif self_byctype is ctype:
            # storing a single ctype
            for child in self.__order.values():
                yield child

    #
    # Interface
    #

    def __setattr__(self, name, obj):
        if name in self._block_reserved_words:
            raise ValueError("Attempting to modify a reserved "
                             "block attribute: %s" % (name,))
        needs_del = False
        same_obj = False
        self_order = self.__order
        if name in self_order:
            # to avoid an edge case, we need to delay
            # deleting the current object until we
            # check the parent of the new object
            needs_del = True
            if self_order[name] is not obj:
                logger.warning(
                    "Implicitly replacing attribute %s (type=%s) "
                    "on block with new object (type=%s). This "
                    "is usually indicative of a modeling error. "
                    "To avoid this warning, delete the original "
                    "object from the block before assigning a new "
                    "object."
                    % (name,
                       getattr(self, name).__class__.__name__,
                       obj.__class__.__name__))
            else:
                same_obj = True
                assert obj.parent is self

        try:
            ctype = obj.ctype
        except AttributeError:
            if needs_del:
                delattr(self, name)
        else:
            if (obj.parent is None) or same_obj:
                if needs_del:
                    delattr(self, name)
                obj._update_parent_and_storage_key(self, name)
                self_order[name] = obj
                self_byctype = self.__byctype
                if self_byctype is None:
                    # storing a single ctype
                    self.__dict__['_block__byctype'] = ctype
                elif self_byctype.__class__ is _ordered_dict_:
                    # large-block storage
                    if ctype not in self_byctype:
                        self_byctype[ctype] = _ordered_dict_()
                    self_byctype[ctype][name] = obj
                elif self_byctype.__class__ is int:
                    # small-block storage
                    # (_byctype is a union of hash bytes)
                    if len(self_order) > self._lbs_count:
                        # activate the large storage format
                        # if we have exceeded the threshold
                        self._activate_large_storage_mode()
                    else:
                        self.__dict__['_block__byctype'] |= hash(ctype)
                else:
                    # currently storing a single ctype
                    if ctype is not self_byctype:
                        if len(self_order) > self._lbs_count:
                            # activate the large storage format
                            # if we have exceeded the threshold
                            self._activate_large_storage_mode()
                        else:
                            self.__dict__['_block__byctype'] = \
                                hash(self_byctype) | hash(ctype)
            else:
                raise ValueError(
                    "Invalid assignment to %s type with name '%s' "
                    "at entry %s. A parent container has already "
                    "been assigned to the object being inserted: %s"
                    % (self.__class__.__name__,
                       self.name,
                       name,
                       obj.parent.name))
        super(block, self).__setattr__(name, obj)

    def __delattr__(self, name):
        self_order = self.__order
        if name in self_order:
            obj = self_order[name]
            del self_order[name]
            obj._clear_parent_and_storage_key()
            self_byctype = self.__byctype
            if self_byctype.__class__ is _ordered_dict_:
                # large-block storage
                ctype = obj.ctype
                del self_byctype[ctype][name]
                if len(self_byctype[ctype]) == 0:
                    del self_byctype[ctype]
        super(block, self).__delattr__(name)

    def write(self,
              filename,
              format=None,
              _solver_capability=None,
              _called_by_solver=False,
              **kwds):
        """
        Write the model to a file, with a given format.

        Args:
            filename (str): The name of the file to write.
            format: The file format to use. If this is not
                specified, the file format will be inferred
                from the filename suffix.
            **kwds: Additional keyword options passed to the
                model writer.

        Returns:
            a :class:`SymbolMap`
        """
        import pyomo.opt
        #
        # Guess the format if none is specified
        #
        if format is None:
            format = pyomo.opt.base.guess_format(filename)
        problem_writer = pyomo.opt.WriterFactory(format)

        if problem_writer is None:
            raise ValueError(
                "Cannot write model in format '%s': no model "
                "writer registered for that format"
                % str(format))

        if _solver_capability is None:
            _solver_capability = lambda x: True
        (filename_, smap) = problem_writer(self,
                                           filename,
                                           _solver_capability,
                                           kwds)
        assert filename_ == filename

        if _called_by_solver:
            # BIG HACK
            smap_id = id(smap)
            if not hasattr(self, "._symbol_maps"):
                setattr(self, "._symbol_maps", {})
            getattr(self, "._symbol_maps")[smap_id] = smap
            return smap_id
        else:
            return smap

    def load_solution(self,
                      solution,
                      allow_consistent_values_for_fixed_vars=False,
                      comparison_tolerance_for_fixed_vars=1e-5):
        """
        Load a solution.

        Args:
            solution: A :class:`pyomo.opt.Solution` object with a
                symbol map. Optionally, the solution can be tagged
                with a default variable value (e.g., 0) that will be
                applied to those variables in the symbol map that do
                not have a value in the solution.
            allow_consistent_values_for_fixed_vars:
                Indicates whether a solution can specify
                consistent values for variables that are
                fixed.
            comparison_tolerance_for_fixed_vars: The
                tolerance used to define whether or not a
                value in the solution is consistent with the
                value of a fixed variable.
        """
        from pyomo.core.kernel.suffix import \
            import_suffix_generator

        symbol_map = solution.symbol_map
        default_variable_value = getattr(solution,
                                         "default_variable_value",
                                         None)

        # Generate the list of active import suffixes on
        # this top level model
        valid_import_suffixes = \
            {obj.storage_key:obj
                 for obj in import_suffix_generator(self)}

        # To ensure that import suffix data gets properly
        # overwritten (e.g., the case where nonzero dual
        # values exist on the suffix and but only sparse
        # dual values exist in the results object) we clear
        # all active import suffixes.
        for suffix in six.itervalues(valid_import_suffixes):
            suffix.clear()

        # Load problem (model) level suffixes. These would
        # only come from ampl interfaced solution suffixes
        # at this point in time.
        for _attr_key, attr_value in six.iteritems(solution.problem):
            attr_key = _attr_key[0].lower() + _attr_key[1:]
            if attr_key in valid_import_suffixes:
                valid_import_suffixes[attr_key][self] = attr_value

        #
        # Load variable data
        #
        from pyomo.core.kernel.variable import IVariable
        for var in self.components(ctype=IVariable):
            var.stale = True
        var_skip_attrs = ['id','canonical_label']
        seen_var_ids = set()
        for label, entry in six.iteritems(solution.variable):
            var = symbol_map.getObject(label)
            if (var is None) or \
               (var is SymbolMap.UnknownSymbol):
                # NOTE: the following is a hack, to handle
                #    the ONE_VAR_CONSTANT variable that is
                #    necessary for the objective
                #    constant-offset terms.  probably should
                #    create a dummy variable in the model
                #    map at the same time the objective
                #    expression is being constructed.
                if "ONE_VAR_CONST" in label:
                    continue
                else:
                    raise KeyError("Variable associated with symbol '%s' "
                                   "is not found on this block"
                                   % (label))

            seen_var_ids.add(id(var))

            if (not allow_consistent_values_for_fixed_vars) and \
               var.fixed:
                raise ValueError("Variable '%s' is currently fixed. "
                                 "A new value is not expected "
                                 "in solution" % (var.name))

            for _attr_key, attr_value in six.iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key == 'value':
                    if allow_consistent_values_for_fixed_vars and \
                       var.fixed and \
                       (math.fabs(attr_value - var.value) > \
                        comparison_tolerance_for_fixed_vars):
                        raise ValueError(
                            "Variable %s is currently fixed. "
                            "A value of '%s' in solution is "
                            "not within tolerance=%s of the current "
                            "value of '%s'"
                            % (var.name, attr_value,
                               comparison_tolerance_for_fixed_vars,
                               var.value))
                    var.value = attr_value
                    var.stale = False
                elif attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][var] = attr_value

        # start to build up the set of unseen variable ids
        unseen_var_ids = set(symbol_map.byObject.keys())
        # at this point it contains ids for non-variable types
        unseen_var_ids.difference_update(seen_var_ids)

        #
        # Load objective solution (should simply be suffixes if
        # they exist)
        #
        objective_skip_attrs = ['id','canonical_label','value']
        for label,entry in six.iteritems(solution.objective):
            obj = symbol_map.getObject(label)
            if (obj is None) or \
               (obj is SymbolMap.UnknownSymbol):
                raise KeyError("Objective associated with symbol '%s' "
                                "is not found on this block"
                                % (label))
            # Because of __default_objective__, an objective might
            # appear twice in the objective dictionary.
            unseen_var_ids.discard(id(obj))
            for _attr_key, attr_value in six.iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][obj] = \
                        attr_value

        #
        # Load constraint solution
        #
        con_skip_attrs = ['id', 'canonical_label']
        for label, entry in six.iteritems(solution.constraint):
            con = symbol_map.getObject(label)
            if con is SymbolMap.UnknownSymbol:
                #
                # This is a hack - see above.
                #
                if "ONE_VAR_CONST" in label:
                    continue
                else:
                    raise KeyError("Constraint associated with symbol '%s' "
                                   "is not found on this block"
                                   % (label))
            unseen_var_ids.remove(id(con))
            for _attr_key, attr_value in six.iteritems(entry):
                attr_key = _attr_key[0].lower() + _attr_key[1:]
                if attr_key in valid_import_suffixes:
                    valid_import_suffixes[attr_key][con] = \
                        attr_value


        #
        # Load sparse variable solution
        #
        if default_variable_value is not None:
            for var_id in unseen_var_ids:
                var = symbol_map.getObject(symbol_map.byObject[var_id])
                if var.ctype is not IVariable:
                    continue
                if (not allow_consistent_values_for_fixed_vars) and \
                   var.fixed:
                    raise ValueError("Variable '%s' is currently fixed. "
                                     "A new value is not expected "
                                     "in solution" % (var.name))

                if allow_consistent_values_for_fixed_vars and \
                   var.fixed and \
                   (math.fabs(default_variable_value - var.value) > \
                    comparison_tolerance_for_fixed_vars):
                    raise ValueError(
                        "Variable %s is currently fixed. "
                        "A value of '%s' in solution is "
                        "not within tolerance=%s of the current "
                        "value of '%s'"
                        % (var.name, default_variable_value,
                           comparison_tolerance_for_fixed_vars,
                           var.value))
                var.value = default_variable_value
                var.stale = False

# inserts class definitions for simple _tuple, _list, and
# _dict containers into this module
define_simple_containers(globals(),
                         "block",
                         IBlock)

# populate the initial set of reserved block attributes so
# that users can not overwrite them when building a model
block._refresh_block_reserved_words()
