#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# This module is meant as a tool for developers. Rarely should a user
# require any functions defined in this module.
#

__all__ = ()

import sys
import copy
import logging
import textwrap

from pyomo.common.config import (ConfigValue,
                                  ConfigBlock)
from pyomo.core.base import maximize, minimize
import argparse

import six

logger = logging.getLogger('pyomo.pysp')

# Major Changes:
#  - Separated scenario tree manager from solver manager
#  - misc renames (boundsetter -> postinit)
#  - deprecate rhosetter?

# FINISHED TODOS:
# - restoreCachedSolutions to restore_cached_solutions
# - cacheSolutions to cache_solutions
# - anything phpyro named to pyro
# - from model_name to model_location
# - objective_sense to objective_sense_stage_based
# - from bound_cfgfile to postinit_callback_location
# - from aggregate_cfgfile to aggregategetter_callback_location
# - from "--scenario-tree-seed" to "--scenario-tree-random-seed"
# - from solver_manager scenario_tree_manager
# - profile_memory implemented?
# - add and implement option to disable PH advanced preprocessing

# TODO:
# - add pyro solver manager support with pyro scenario tree manager
# - implement ph_timelimit
# - generalize for options configurations with enable_ww_extensions, ww_extension_cfgfile, ww_extension_annotationfile, user_defined_extension,
# - generalize options collection for ph convergers

# Maybe TODOS:
# - from rho_cfgfile to phrhosetter_callback_location
# - Default True? for enable_normalized_termdiff_convergence
# - implementation of drop_proximal_terms

#
# Subclassing the pyutilib versions just
# in case we need to add functionality
#
class PySPConfigValue(ConfigValue):
    __slots__ = ('ap_group','ap_kwds','ap_args')

    def __init__(self, *args, **kwds):
        super(PySPConfigValue, self).__init__(*args, **kwds)
        self.ap_group = None
        self.ap_args = ()
        self.ap_kwds = {}

    def __getstate__(self):
        ans = super(PySPConfigValue, self).__getstate__()
        for key in PySPConfigValue.__slots__:
            ans[key] = getattr(self, key)
        return ans

    def __str__(self):
        return ("%s(name=%s)" % (self.__class__.__name__,
                                 self._name))
    __repr__ = __str__

class PySPConfigBlock(ConfigBlock):

    def about(self, name):
        """Return a summary string for an option
        registered on this block."""
        configval = self.get(name)
        outstr = ("%s: %s\n"
                  % (configval.__class__.__name__,
                     configval._name))
        outstr += ("  -    type: %s\n"
                   "  - default: %s\n"
                   % ((configval._domain.doc if \
                       hasattr(configval._domain, 'doc') else \
                       configval._domain),
                      configval._default))
        if configval._description is None:
            lines = ["None"]
        else:
            lines = textwrap.wrap(configval._description,
                                  width=58)
        while lines[-1].strip() == "":
            lines.pop()
        for i, line in enumerate(lines):
            if i == 0:
                tmp = "  -    doc: %s"
            else:
                tmp = "            %s"
            if i != len(lines) - 1:
                tmp += "\n"
            tmp %= line
            outstr += tmp

        return outstr

    #
    # Overriding the behavior of ConfigBlock.display
    # so that it prints by default. I think it is
    # confusing that a method on that class named
    # 'display' returns a string
    #
    def display(self, ostream=None, **kwds):
        """Displays the list of options registered to this
        block. The optional keyword 'ostream' can be a file
        like object to write to."""
        # Note: this functionality has been migrated into the
        # PyUtilib.ConfigBlock.  We will attempt the new API and fall
        # back on the old one in case PyUtilib is too old.
        try:
            super(PySPConfigBlock, self).display(ostream=ostream, **kwds)
        except TypeError:
            outstr = super(PySPConfigBlock, self).display(**kwds)
            if ostream is None:
                print(outstr)
            else:
                ostream.write(outstr)

    #
    # Change the default behavior of ConfigBlock when
    # _implicit_declaration are allowed. For some reason, in
    # this case underscores are removed in implicitly
    # assigned names (which does not happen when __setitem__
    # is called).
    #
    def __setattr__(self, name, value):
        # this is copied from the base class, except the
        # part that replaces '_' with ' ' is removed
        if name in ConfigBlock._all_slots:
            super(ConfigBlock, self).__setattr__(name, value)
        else:
            ConfigBlock.__setitem__(self, name, value)

    #
    # Allow deletion of entries to get around issues with
    # clashing options registration
    #
    def __delattr__(self, name):
        if name not in self._data:
            _name = name.replace('_', ' ')
            if _name not in self._data:
                raise AttributeError("Unknown attribute '%s'" % name)
            name = _name
        self._data[name]._parent = None
        del self._data[name]
        self._decl_order.remove(name)

    #
    # Change more strange default behavior for ConfigBlock
    # when _implicit_declaration are allowed. If the userSet
    # flag gets set to True on ConfigBlock (happens when new
    # implicit entry is assigned), this will cause strange
    # behavior with the user_values() method (it returns
    # this block).
    #
    # Also, make sure that implicitly created entries
    # have their _userSet flag set to True.
    def add(self, name, config):
        ans = super(PySPConfigBlock, self).add(name, config)
        ans._userSet = True
        self._userSet = False
        return ans

    def check_usage(self, error=True):
        """Check for usage of options that have set
        by a user.

        Args:
            error (bool): When :const:`True` (default), an
                exception is raised when any options set by
                the user have not been accessed by any
                code. Otherwise, a warning is logged.

        Returns:
            :const:`True` when all options set by the user \
            have been accessed; otherwise, returns \
            :const:`False` if the :attr:`error` keyword is
            set to :const:`False`.

        Raises:
            ValueError: when user-set options exist but have
                not been accessed (and the :attr:`error`
                keyword is :const:`True`)
        """
        ignored_options = dict((_c._name, _c.value(False))
                               for _c in self.unused_user_values())
        if len(ignored_options):
            msg = ("The following options were "
                   "explicitly set but never accessed:\n")
            for name in sorted(ignored_options):
                msg += (" - %s: %s\n" % (name, ignored_options[name]))
            msg += ("If you believe this is a bug, please report it "
                    "to the PySP developers.\n")
            logger.warning(msg)
            if error:
                raise ValueError(msg)
            else:
                return False
        return True


def check_options_match(opt1,
                        opt2,
                        include_argparse=True,
                        include_default=True,
                        include_value=True,
                        include_accessed=True):
    assert isinstance(opt1, PySPConfigValue)
    assert isinstance(opt2, PySPConfigValue)
    if (opt1._domain == opt2._domain) and \
       (opt1._description == opt2._description) and \
       (opt1._doc == opt2._doc) and \
       (opt1._visibility == opt2._visibility) and \
       ((not include_argparse) or (opt1._argparse == opt2._argparse)) and \
       ((not include_default) or (opt1._default == opt2._default)) and \
       ((not include_value) or (opt1._data == opt2._data)) and \
       ((not include_value) or (opt1._userSet == opt2._userSet)) and \
       ((not include_accessed) or (opt1._userAccessed == opt2._userAccessed)):
        return

    msg = "Options do not match. This is likely a developer error. Summary:\n"
    if opt1._domain != opt2._domain:
        msg += "\n"
        msg += "opt1._domain: "+str(opt1._domain)+"\n"
        msg += "opt2._domain: "+str(opt2._domain)+"\n"
    if opt1._description != opt2._description:
        msg += "\n"
        msg += "opt1._description: "+str(opt1._description)+"\n"
        msg += "opt2._description: "+str(opt2._description)+"\n"
    if opt1._doc != opt2._doc:
        msg += "\n"
        msg += "opt1._doc: "+str(opt1._doc)+"\n"
        msg += "opt2._doc: "+str(opt2._doc)+"\n"
    if opt1._visibility != opt2._visibility:
        msg += "\n"
        msg += "opt1._visibility: "+str(opt1._visibility)+"\n"
        msg += "opt2._visibility: "+str(opt2._visibility)+"\n"
    if (include_argparse and (opt1._argparse != opt2._argparse)):
        msg += "\n"
        msg += "opt1._argparse: "+str(opt1._argparse)+"\n"
        msg += "opt2._argparse: "+str(opt2._argparse)+"\n"
    if (include_default and (opt1._default != opt2._default)):
        msg += "\n"
        msg += "opt1._default: "+str(opt1._default)+"\n"
        msg += "opt2._default: "+str(opt2._default)+"\n"
    if (include_value and (opt1._data != opt2._data)):
        msg += "\n"
        msg += "opt1._data: "+str(opt1._data)+"\n"
        msg += "opt2._data: "+str(opt2._data)+"\n"
    if (include_value and (opt1._userSet != opt2._userSet)):
        msg += "\n"
        msg += "opt1._userSet: "+str(opt1._userSet)+"\n"
        msg += "opt2._userSet: "+str(opt2._userSet)+"\n"
    if (include_accessed and (opt1._userAccessed != opt2._userAccessed)):
        msg += "\n"
        msg += "opt1._userAccessed: "+str(opt1._userAccessed)+"\n"
        msg += "opt2._userAccessed: "+str(opt2._userAccessed)+"\n"
    raise ValueError(msg)

#
# register an option to a PySPConfigBlock,
# making sure nothing is overwritten
#
def safe_declare_option(configblock,
                        name,
                        configvalue,
                        ap_group=None,
                        relax_default_check=False,
                        declare_for_argparse=False,
                        ap_args=None,
                        ap_kwds=None):
    assert isinstance(configblock, PySPConfigBlock)
    assert configvalue._parent == None
    assert configvalue._userSet == False
    assert configvalue._userAccessed == False
    if ap_group is not None:
        assert isinstance(ap_group, six.string_types)
        configvalue.ap_group = ap_group
    if ap_args is not None:
        assert type(ap_args) is tuple
        configvalue.ap_args = ap_args
    if ap_kwds is not None:
        assert type(ap_kwds) is dict
        configvalue.ap_kwds = ap_kwds
    if name not in configblock:
        configblock.declare(
            name,
            copy.deepcopy(configvalue))
        if declare_for_argparse:
            assert configblock.get(name)._argparse is None
            if ap_args is None:
                ap_args = configblock.get(name).ap_args
            if ap_kwds is None:
                ap_kwds = configblock.get(name).ap_kwds
            if (configblock.get(name).ap_group is not None) and \
               ('group' not in ap_kwds):
                ap_kwds['group'] = configblock.get(name).ap_group
            configblock.get(name).\
                declare_as_argument(*ap_args, **ap_kwds)
    else:
        current = configblock.get(name)
        check_options_match(current,
                            configvalue,
                            include_default=not relax_default_check,
                            include_argparse=False,
                            include_value=False,
                            include_accessed=False)
        if declare_for_argparse:
            assert current._argparse is None
            if ap_args is None:
                ap_args = current.ap_args
            if ap_kwds is None:
                ap_kwds = current.ap_kwds
            if (current.ap_group is not None) and \
               ('group' not in ap_kwds):
                ap_kwds['group'] = current.ap_group
            current.declare_as_argument(*ap_args, **ap_kwds)

#
# Register an option to a PySPConfigBlock,
# throwing an error if the name is not new
#
def safe_declare_unique_option(configblock,
                               name,
                               configvalue,
                               ap_group=None,
                               declare_for_argparse=False,
                               ap_args=None,
                               ap_kwds=None,
                               check_for_deprecated=False):
    assert isinstance(configblock, PySPConfigBlock)
    assert configvalue._parent == None
    assert configvalue._userSet == False
    assert configvalue._userAccessed == False
    if name in configblock:
        raise RuntimeError(
            "Option registration failed. An option "
            "with name '%s' already exists on the PySPConfigBlock."
            % (name))
    if ap_group is not None:
        assert isinstance(ap_group, six.string_types)
        configvalue.ap_group = ap_group
    if ap_args is not None:
        assert type(ap_args) is tuple
        configvalue.ap_args = ap_args
    if ap_kwds is not None:
        assert type(ap_kwds) is dict
        configvalue.ap_kwds = ap_kwds
    configblock.declare(
        name,
        copy.deepcopy(configvalue))
    assert configblock.get(name)._argparse is None
    if declare_for_argparse:
        if ap_args is None:
            ap_args = configblock.get(name).ap_args
        if ap_kwds is None:
            ap_kwds = configblock.get(name).ap_kwds
        if (configblock.get(name).ap_group is not None) and \
           ('group' not in ap_kwds):
            ap_kwds['group'] = configblock.get(name).ap_group
        configblock.get(name).\
            declare_as_argument(*ap_args, **ap_kwds)
        assert not check_for_deprecated
    else:
        if check_for_deprecated:
            if name in _map_to_deprecated:
                deprecated_value = _map_to_deprecated[name]
                assert deprecated_value._parent == _deprecated_block
                assert deprecated_value._userSet == False
                assert deprecated_value._userAccessed == False
                deprecated_value._parent = None
                deprecated_value_copy = copy.deepcopy(deprecated_value)
                deprecated_value._parent = _deprecated_block
                configblock.declare(deprecated_value_copy._name,
                                    deprecated_value_copy)

#
# Register an option to a PySPConfigBlock,
# throwing an error if the name is not new.
# After registering the option, make sure
# it has been declared for argparse
#
def safe_register_unique_option(configblock,
                                name,
                                configvalue,
                                ap_group=None,
                                ap_args=None,
                                ap_kwds=None):
    safe_declare_unique_option(configblock,
                               name,
                               configvalue,
                               ap_group=ap_group,
                               ap_args=ap_args,
                               ap_kwds=ap_kwds,
                               declare_for_argparse=True)

common_block = PySPConfigBlock("A collection of common PySP options")

def _domain_unit_interval(val):
    val = float(val)
    if not (0 <= val <= 1):
        raise ValueError(
            "Value %s is not in the interval [0,1]."
            % (val))
    return val
_domain_unit_interval.doc = \
    "<domain: value in the interval [0,1]>"

def _domain_percent(val):
    val = float(val)
    if not (0 <= val <= 100):
        raise ValueError(
            "Value %s is not between 0 and 100")
    return val
_domain_percent.doc = \
    "<domain: value in the interval [0,100]>"

def _domain_nonnegative_integer(val):
    val = int(val)
    if val < 0:
        raise ValueError(
            "Value %s is not a non-negative integer."
            % (val))
    return val
_domain_nonnegative_integer.doc = \
    "<domain: non-negative integer>"

def _domain_positive_integer(val):
    val = int(val)
    if val <= 0:
        raise ValueError(
            "Value %s is not a positive integer."
            % (val))
    return val
_domain_positive_integer.doc = \
    "<domain: positive integer>"

class _domain_integer_interval(object):
    def __init__(self, start, stop):
        assert start <= stop
        assert int(start) == start
        assert int(stop) == stop
        self.start = int(start)
        self.stop = int(stop)
        self.doc = ("<domain: integer interval [%d, %d]"
                    % (self.start, self.stop))
    def __call__(self, val):
        val = int(val)
        if not (self.start <= val <= self.stop):
            raise ValueError(
                "Value %s is not in integer interval [%d, %d]."
                % (val, self.start, self.stop))
        return val

def _domain_nonnegative(val):
    val = float(val)
    if val < 0:
        raise ValueError(
            "Value %s is not non-negative."
            % (val))
    return val
_domain_nonnegative.doc = \
    "<domain: non-negative value>"

def _domain_positive(val):
    val = float(val)
    if val <= 0:
        raise ValueError(
            "Value %s is not positive."
            % (val))
    return val
_domain_nonnegative.doc = \
    "<domain: non-negative value>"

def _domain_must_be_str(val):
    if not isinstance(val, six.string_types):
        raise TypeError(
            "Value must be a built-in "
            "string type, not '%s'" % (type(val)))
    return val
_domain_must_be_str.doc = \
    "<domain: string>"

def _domain_tuple_of_str(val):
    if isinstance(val, six.string_types):
        return (val,)
    elif not isinstance(val, (list, tuple)):
        raise TypeError(
            "Value must be a built-in list or "
            "tuple of string types, not '%s'" % (type(val)))
    else:
        for _v in val:
            if not isinstance(_v, six.string_types):
                raise TypeError(
                    "Value must be a built-in "
                    "string type, not '%s'" % (type(_v)))
        return tuple(_v for _v in val)
_domain_tuple_of_str.doc = \
    "<domain: list of strings>"

def _domain_tuple_of_str_or_dict(val):
    if isinstance(val, six.string_types):
        return (val,)
    elif isinstance(val, (list, tuple)):
        for _v in val:
            if not isinstance(_v, six.string_types):
                raise TypeError(
                    "Value must be a built-in "
                    "string type, not '%s'" % (type(_v)))
        return tuple(_v for _v in val)
    elif isinstance(val, dict):
        for _v in val:
            if not isinstance(_v, six.string_types):
                raise TypeError(
                    "Dict keys must be a built-in "
                    "string type, not '%s'" % (type(_v)))
        return val
    else:
        raise TypeError(
            "Value must be a built-in list or "
            "tuple of string types, or a dict, not '%s'" % (type(val)))
_domain_tuple_of_str_or_dict.doc = \
    "<domain: dict or list of strings>"

#
# Common 'Input Options'
#
_input_options_group_title = "Input Options"

safe_declare_unique_option(
    common_block,
    "model_location",
    PySPConfigValue(
        ".",
        domain=_domain_must_be_str,
        description=(
            "The directory or filename where the reference model is "
            "found. If a directory is given, the reference model is "
            "assumed to reside in a file named 'ReferenceModel.py' under "
            "that directory. If the reference model file contains a "
            "function named 'pysp_instance_creation_callback', it will "
            "be called to construct each scenario model (not necessarily "
            "by the same running process). Otherwise, the reference "
            "model file must contain an AbstractModel object named "
            "'model'. Default is '.'. "
        ),
        doc=None,
        visibility=0),
    ap_args=("-m", "--model-location"),
    ap_group=_input_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "scenario_tree_location",
    PySPConfigValue(
        None,
        domain=_domain_must_be_str,
        description=(
            "The directory or filename where the scenario tree "
            "structure .dat file is found. If a directory is "
            "given, the scenario tree structure is assumed to "
            "reside in a file named 'ScenarioStructure.dat' under "
            "that directory. The location of any required scenario "
            "data files will also be inferred from the value of "
            "this option. If unspecified, the reference model file "
            "must define a function named "
            "'pysp_scenario_tree_model_callback' and must be in the "
            "same location as any required scenario data files."
        ),
        doc=None,
        visibility=0),
    ap_args=("-s", "--scenario-tree-location"),
    ap_group=_input_options_group_title,
    declare_for_argparse=True)

_objective_sense_choices = \
    [maximize, 'max', 'maximize',
     minimize, 'min', 'minimize', None]
def _objective_sense_domain(val):
    if val in ('min', 'minimize', minimize):
        return minimize
    elif val in ('max', 'maximize', maximize):
        return maximize
    elif val is None:
        return None
    else:
        raise ValueError(
            "Invalid choice: %s. (choose from one of %s"
            % (val, _objective_sense_choices))
safe_declare_unique_option(
    common_block,
    "objective_sense_stage_based",
    PySPConfigValue(
        None,
        domain=_objective_sense_domain,
        description=(
            "The objective sense to use when auto-generating the "
            "scenario instance objective function, which is equal to "
            "the sum of the scenario-tree stage costs declared on the "
            "reference model.  If unspecified, it is assumed a "
            "stage-cost based objective function has been declared on "
            "the reference model."
        ),
        doc=None,
        visibility=0),
    ap_args=("-o", "--objective-sense-stage-based"),
    ap_kwds={'choices':_objective_sense_choices},
    ap_group=_input_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "postinit_callback_location",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str,
        description=(
            "File that contains a 'pysp_postinit_callback' "
            "function, which is executed on each scenario at "
            "the end of scenario tree manager "
            "initialization. If the scenario tree is "
            "distributed, then this callback will be "
            "transmitted to the respective scenario tree "
            "workers where the constructed scenario instances "
            "are available. This callback can be used to "
            "update things like variable bounds as well as "
            "other scenario-specific information stored on "
            "the Scenario objects. This callback will be "
            "executed immediately after any "
            "'pysp_aggregategetter_callback' function that is "
            "specified. This option can be used multiple "
            "times from the command line to specify more than "
            "one callback function location."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_input_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "aggregategetter_callback_location",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str,
        description=(
            "File that contains a "
            "'pysp_aggregategetter_callback' function, which "
            "is executed in a sequential call chain on each "
            "scenario at the end of scenario tree manager "
            "initialization. Most useful in cases where the "
            "scenario tree is distributed across multiple "
            "processes, it can be used to execute arbitrary "
            "code whose return value is passed as input into "
            "the next call in the chain. At the end of the "
            "call chain, the final result is broadcast to all "
            "scenario tree worker processes and stored under "
            "the name _aggregate_user_data on the worker "
            "object. Potential uses include collecting "
            "aggregate scenario information that is "
            "subsequently used by a 'pysp_postinit_callback' "
            "function to set tight variable bounds. This "
            "option can be used multiple times from the "
            "command line to specify more than one callback "
            "function location."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_input_options_group_title,
    declare_for_argparse=True)

#
# Common 'Scenario Tree Options'
#
_scenario_tree_options_group_title = "Scenario Tree Options"

safe_declare_unique_option(
    common_block,
    "scenario_tree_random_seed",
    PySPConfigValue(
        None,
        domain=int,
        description=(
            "The random seed associated with manipulation operations "
            "on the scenario tree (e.g., down-sampling or bundle "
            "creation). Default is None, indicating unassigned."
        ),
        doc=None,
        visibility=0),
    ap_group=_scenario_tree_options_group_title)

safe_declare_unique_option(
    common_block,
    "scenario_tree_downsample_fraction",
    PySPConfigValue(
        1,
        domain=_domain_unit_interval,
        description=(
            "The proportion of the scenarios in the scenario tree that "
            "are actually used.  Specific scenarios are selected at "
            "random.  Default is 1.0, indicating no down-sampling."
        ),
        doc=None,
        visibility=0),
    ap_group=_scenario_tree_options_group_title)

safe_declare_unique_option(
    common_block,
    "scenario_bundle_specification",
    PySPConfigValue(
        None,
        domain=None,
        description=(
            "The name of the scenario bundling specification to be "
            "used when generating the scenario tree. Default is "
            "None, indicating no bundling is employed. If the "
            "specified name ends with a .dat suffix, the argument is "
            "interpreted as the path to a file. Otherwise, the name "
            "is interpreted as a file in the instance directory, "
            "constructed by adding the .dat suffix automatically. "
            "If scripting, this option can alternatively be assigned "
            "a dictionary mapping bundle names to a list of scenario "
            "names."
        ),
        doc=None,
        visibility=0),
    ap_group=_scenario_tree_options_group_title)

safe_declare_unique_option(
    common_block,
    "create_random_bundles",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "Specification to create the indicated number of random, "
            "equally-sized (to the degree possible) scenario "
            "bundles. Default is 0, indicating no scenario bundles "
            "will be created."
        ),
        doc=None,
        visibility=0),
    ap_group=_scenario_tree_options_group_title)

safe_declare_unique_option(
    common_block,
    "scenario_tree_manager",
    PySPConfigValue(
        "serial",
        domain=_domain_must_be_str,
        description=(
            "The type of scenario tree manager to use. The default, "
            "'serial', builds all scenario instances on the parent "
            "process and performs all scenario tree operations "
            "sequentially. If 'pyro' is specified, the scenario tree "
            "is fully distributed and scenario tree operations are "
            "performed asynchronously."
        ),
        doc=None,
        visibility=0),
    ap_group=_scenario_tree_options_group_title)

#
# Common 'Pyro Options'
#
_pyro_options_group_title = "Pyro Options"

safe_declare_unique_option(
    common_block,
    "pyro_host",
    PySPConfigValue(
        None,
        domain=_domain_must_be_str,
        description=(
            "The hostname to bind on when searching for a Pyro "
            "nameserver."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)

safe_declare_unique_option(
    common_block,
    "pyro_port",
    PySPConfigValue(
        None,
        domain=int,
        description=(
            "The port to bind on when searching for a Pyro "
            "nameserver."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)


safe_declare_unique_option(
    common_block,
    "pyro_handshake_at_startup",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Take extra steps to acknowledge Pyro-based requests are "
            "received by workers during pyro scenario tree manager "
            "initialization. This option can be useful for debugging "
            "connection issues during startup."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)

safe_declare_unique_option(
    common_block,
    "pyro_required_scenariotreeservers",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "Set the number of idle scenario tree server processes "
            "expected to be available when the 'pyro' scenario tree "
            "manager is selected. This option should be used when the "
            "number of workers is less than the total number of "
            "scenarios (or bundles). The default value of 0 "
            "indicates that the manager should attempt to assign each "
            "scenario (or bundle) to a single scenariotreeserver process "
            "until the timeout (indicated by the "
            "pyro_find_scenariotreeservers_timeout option) occurs."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)

safe_declare_unique_option(
    common_block,
    "pyro_find_scenariotreeservers_timeout",
    PySPConfigValue(
        30,
        domain=float,
        description=(
            "Set the time limit (seconds) for finding idle "
            "scenario tree server processes when the 'pyro' "
            "scenario tree manager is selected. This option "
            "is ignored when "
            "pyro_required_scenariotreeservers is used. "
            "Default is 30 seconds."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)

safe_declare_unique_option(
    common_block,
    "pyro_shutdown",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Attempt to shut down all Pyro-related components "
            "associated with the Pyro name server for the "
            "scenario tree manager. Components to shutdown "
            "include the name server, dispatch server, and "
            "any scenariotreeserver registered with the "
            "dispatcher. Note that if Pyro4 is in use the "
            "nameserver will always ignore this request."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)

safe_declare_unique_option(
    common_block,
    "pyro_shutdown_workers",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Upon exit, send shutdown requests to all worker "
            "processes that were acquired through the "
            "dispatcher used by the scenario tree manager. "
            "This leaves any dispatchers and namservers "
            "running as well as any processes registered with "
            "the dispather that were not acquired for work by "
            "this client."
        ),
        doc=None,
        visibility=0),
    ap_group=_pyro_options_group_title)

#
# Common 'Output Options'
#
_output_options_group_title = "Output Options"

safe_declare_unique_option(
    common_block,
    "output_solver_results",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Output solutions obtained after each scenario sub-problem "
            "solve."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "output_scenario_tree_solution",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Report the solution in scenario tree format upon termination. "
            "Default is False."
            ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "symbolic_solver_labels",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "When interfacing with the solver, use symbol names "
            "derived from the model. For example, "
            "'my_special_variable[1_2_3]' instead of 'x552'.  Useful "
            "for debugging. When using NL file based solvers, this "
            "option results in corresponding .row (constraints) and "
            ".col (variables) file being created. The ordering in these "
            "files provides a mapping from NL file index to symbolic "
            "model names."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "file_determinism",
    PySPConfigValue(
        1,
        domain=int,
        description=(
            "When interfacing with a solver using file based I/O, set "
            "the effort level for ensuring the file creation process is "
            "determistic. The default (1) sorts the index of components "
            "when transforming the model.  Anything less than 1 "
            "disables index sorting and can speed up model I/O. "
            "Anything greater than 1 additionaly sorts by component "
            "name to override declartion order."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "output_solver_log",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Output solver log during solves."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "output_times",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Output timing statistics during various runtime stages."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "output_instance_construction_time",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Output timing information during instance construction. "
            "This option will be ignored when a "
            "'pysp_instance_creation_callback' function is defined "
            "inside the reference model file."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "keep_solver_files",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Retain temporary input and output files for scenario "
            "sub-problem solves."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

safe_declare_unique_option(
    common_block,
    "verbose",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Generate verbose output for both initialization and "
            "execution."
        ),
        doc=None,
        visibility=0),
    ap_group=_output_options_group_title)

#
# Common 'PH Options'
#
_ph_options_group_title = "PH Options"

safe_declare_unique_option(
    common_block,
    "ph_warmstart_file",
    PySPConfigValue(
        "",
        domain=_domain_must_be_str,
        description=(
            "Disable iteration 0 solves and warmstart rho, weight, "
            "and xbar parameters from solution or history file."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "ph_warmstart_index",
    PySPConfigValue(
        "",
        domain=_domain_must_be_str,
        description=(
            "Indicates the iteration inside a history file from which "
            "to load a warmstart."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

def _rho_domain(val):
    val = float(val)
    if val < 0:
        raise ValueError(
            "Invalid value for default rho: %s. "
            "Value must be non-negative or None."
            % (val))
    return val

safe_declare_unique_option(
    common_block,
    "default_rho",
    PySPConfigValue(
        None,
        domain=_rho_domain,
        description=(
            "The default PH rho value for all non-anticipative "
            "variables. *** Required ***"
        ),
        doc=None,
        visibility=0),
    ap_args=("-r", "--default-rho"),
    ap_group=_ph_options_group_title,
    declare_for_argparse=True)

_xhat_method_choices = \
    ['closest-scenario','voting','rounding']
def _xhat_method_domain(val):
    if val in _xhat_method_choices:
        return val
    else:
        raise ValueError(
            "Invalid choice: %s. (choose from one of %s"
            % (val, _xhat_method_choices))

safe_declare_unique_option(
    common_block,
    "xhat_method",
    PySPConfigValue(
        "closest-scenario",
        domain=_xhat_method_domain,
        description=(
            "Specify the method used to compute a bounding solution at "
            "PH termination. Defaults to 'closest-scenario'. Other "
            "variants are: 'voting' and 'rounding'."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'choices':_xhat_method_choices},
    ap_group=_ph_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "overrelax",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Compute weight updates using combination of previous and "
            "current variable averages."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "nu",
    PySPConfigValue(
        1.5,
        domain=float,
        description=(
            "Parameter used to update weights when using the overrelax "
            "option."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "async",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Run PH in asychronous mode after iteration 0."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "async_buffer_length",
    PySPConfigValue(
        1,
        domain=_domain_positive_integer,
        description=(
            "Number of scenarios to collect, if in async mode, before "
            "doing statistics and weight updates. Default is 1."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

#safe_declare_unique_option(
#    common_block,
#    "phrhosetter_callback_location",
#    PySPConfigValue(
#        None,
#        domain=_domain_must_be_str,
#        description=(
#            "File containing a 'pysp_phrhosetter_callback' function, which "
#            "is used to update per-variable rho parameters. This callback "
#            "will be executed during PH initialization."
#        ),
#        doc=None,
#        visibility=0))

safe_declare_unique_option(
    common_block,
    "max_iterations",
    PySPConfigValue(
        100,
        domain=_domain_nonnegative_integer,
        description=(
            "The maximal number of PH iterations. Default is 100."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "ph_timelimit",
    PySPConfigValue(
        None,
        domain=float,
        description=(
            "Limits the number of seconds spent inside the solve "
            "method of PH."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "termdiff_threshold",
    PySPConfigValue(
        0.0001,
        domain=float,
        description=(
            "The convergence threshold used in the term-diff and "
            "normalized term-diff convergence criteria. Default is "
            "0.0001."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "enable_free_discrete_count_convergence",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Terminate PH based on the free discrete variable count "
            "convergence metric."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "free_discrete_count_threshold",
    PySPConfigValue(
        20,
        domain=_domain_positive_integer,
        description=(
            "The convergence threshold used in the criterion based on "
            "when the free discrete variable count convergence "
            "criterion. Default is 20."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "enable_normalized_termdiff_convergence",
    PySPConfigValue(
        True,
        domain=bool,
        description=(
            "Terminate PH based on the normalized termdiff convergence "
            "metric. Default is True. "
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "enable_termdiff_convergence",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Terminate PH based on the termdiff convergence metric."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "enable_outer_bound_convergence",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Terminate PH based on the outer bound convergence "
            "metric."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "outer_bound_convergence_threshold",
    PySPConfigValue(
        None,
        domain=float,
        description=(
            "The convergence threshold used in the outer bound "
            "convergence criterion. Default is None, indicating "
            "unassigned."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "linearize_nonbinary_penalty_terms",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "Approximate the PH quadratic term for non-binary "
            "variables with a piece-wise linear function, using the "
            "supplied number of equal-length pieces from each bound to "
            "the average. The default value of 0 indications no "
            "linearization shall take place."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "breakpoint_strategy",
    PySPConfigValue(
        0,
        domain=int,
        description=(
            "Specify the strategy to distribute breakpoints on the "
            "[lb, ub] interval of each variable when linearizing. 0 "
            "indicates uniform distribution. 1 indicates breakpoints at "
            "the node min and max, uniformly in-between. 2 indicates "
            "more aggressive concentration of breakpoints near the "
            "observed node min/max."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "retain_quadratic_binary_terms",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Do not linearize PH objective terms involving binary "
            "decision variables."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "drop_proximal_terms",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Eliminate proximal terms (i.e., the quadratic penalty "
            "terms) from the weighted PH objective."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "report_solutions",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Always report PH solutions after each iteration. Enabled "
            "if verbose output is enabled."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "report_weights",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Always report PH weights prior to each iteration. Enabled "
            "if verbose output is enabled."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "report_rhos_each_iteration",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Always report PH rhos prior to each iteration."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "report_rhos_first_iteration",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Report rhos prior to PH iteration 1. Enabled if --verbose "
            "is enabled."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "report_for_zero_variable_values",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Report statistics (variables and weights) for all "
            "variables, not just those with values differing from 0."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "report_only_nonconverged_variables",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Report statistics (variables and weights) only for "
            "non-converged variables."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

safe_declare_unique_option(
    common_block,
    "suppress_continuous_variable_output",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Eliminate PH-related output involving continuous "
            "variables."
        ),
        doc=None,
        visibility=0),
    ap_group=_ph_options_group_title)

#
# Common 'Extension Options'
#
_extension_options_group_title = "Extension Options"

safe_declare_unique_option(
    common_block,
    "enable_ww_extensions",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Enable the Watson-Woodruff PH extensions plugin."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "ww_extension_cfgfile",
    PySPConfigValue(
        "",
        domain=_domain_must_be_str,
        description=(
            "The name of a configuration file for the Watson-Woodruff "
            "PH extensions plugin."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "ww_extension_annotationfile",
    PySPConfigValue(
        "",
        domain=_domain_must_be_str,
        description=(
            "The name of a variable annotation file for the "
            "Watson-Woodruff PH extensions plugin."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "user_defined_extension",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str,
        description=(
            "The name of a python module specifying a user-defined PH "
            "extension plugin. Use this option when generating a "
            "template configuration file or invoking command-line help "
            "in order to include any plugin-specific options. This "
            "option can used multiple times from the command line to "
            "specify more than one plugin."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_extension_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "solution_writer",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str,
        description=(
            "The name of a python module specifying a user-defined "
            "plugin invoked to write the scenario tree solution. Use "
            "this option when generating a template configuration file "
            "or invoking command-line help in order to include any "
            "plugin-specific options. This option can be used multiple "
            "times from the command line to specify more than one plugin."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_extension_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "extension_precedence",
    PySPConfigValue(
        0,
        domain=int,
        description=(
            "Sets the priority for execution of this extension "
            "relative to other extensions of the same type. Extensions "
            "with higher precedence values are guaranteed to be "
            "executed before any extensions having strictly lower "
            "precedence values. Default is 0."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "output_name",
    PySPConfigValue(
        None,
        domain=_domain_must_be_str,
        description=(
            "The directory or filename where the scenario tree solution "
            "should be saved to."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "input_name",
    PySPConfigValue(
        None,
        domain=_domain_must_be_str,
        description=(
            "The directory or filename where the scenario tree solution "
            "should be loaded from."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "solution_saver_extension",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str,
        description=(
            "The name of a python module specifying a user-defined "
            "plugin implementing the IPySPSolutionSaverExtension "
            "interface. Invoked to save a scenario tree solution. Use "
            "this option when generating a template configuration file "
            "or invoking command-line help in order to include any "
            "plugin-specific options. This option can be used multiple "
            "times from the command line to specify more than one plugin."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_extension_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "solution_loader_extension",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str,
        description=(
            "The name of a python module specifying a user-defined "
            "plugin implementing the IPySPSolutionLoaderExtension "
            "interface. Invoked to load a scenario tree solution. Use "
            "this option when generating a template configuration file "
            "or invoking command-line help in order to include any "
            "plugin-specific options. This option can be used multiple "
            "times from the command line to specify more than one plugin."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_extension_options_group_title,
    declare_for_argparse=True)

safe_declare_unique_option(
    common_block,
    "save_stages",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "The number of scenario tree time stages to save from the solution. "
            "The default value of 0 indicates that all stages should be saved."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

safe_declare_unique_option(
    common_block,
    "load_stages",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "The number of scenario tree stages to load from the solution. "
            "The default value of 0 indicates that all stages should be loaded."
        ),
        doc=None,
        visibility=0),
    ap_group=_extension_options_group_title)

#
# Common 'Advanced Options'
#
_advanced_options_group_title = 'Advanced Options'

safe_declare_unique_option(
    common_block,
    "pyro_transmit_leaf_stage_variable_solutions",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "By default, when running PH using the pyro scenario "
            "tree manager, leaf-stage variable solutions are not "
            "transmitted back to the master scenario tree during "
            "intermediate iterations. This flag will override that "
            "behavior for cases where leaf-stage variable solutions are "
            "required on the master scenario tree. Using this option "
            "can degrade runtime performance. When PH exits, variable "
            "values are collected from all stages whether or not this "
            "option was used. Also, note that PH extensions have the "
            "ability to override this flag at runtime."
        ),
        doc=None,
        visibility=0),
    ap_group=_advanced_options_group_title)

safe_declare_unique_option(
    common_block,
    "disable_advanced_preprocessing",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Disable advanced preprocessing directives designed to "
            "speed up model I/O for scenario or bundle instances. This "
            "can be useful in debugging situations but will slow down "
            "algorithms that repeatedly solve subproblems. Use of this "
            "option will cause the '--preprocess-fixed-variables' option "
            "to be ignored."
        ),
        doc=None,
        visibility=0),
    ap_group=_advanced_options_group_title)

safe_declare_unique_option(
    common_block,
    "preprocess_fixed_variables",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Perform full preprocessing of instances after fixing or "
            "freeing variables in scenarios. By default, fixed "
            "variables will be included in the problem but 'fixed' by "
            "overriding their bounds.  This increases the speed of "
            "Pyomo model I/O, but may be useful to disable in "
            "debugging situations or if numerical issues are "
            "encountered with certain solvers."
        ),
        doc=None,
        visibility=0),
    ap_group=_advanced_options_group_title)

safe_declare_unique_option(
    common_block,
    "comparison_tolerance_for_fixed_variables",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Perform full preprocessing of instances after fixing or "
            "freeing variables in scenarios. By default, fixed "
            "variables will be included in the problem but 'fixed' by "
            "overriding their bounds.  This increases the speed of "
            "Pyomo model I/O, but may be useful to disable in "
            "debugging situations or if numerical issues are "
            "encountered with certain solvers."
        ),
        doc=None,
        visibility=0),
    ap_group=_advanced_options_group_title)

#
# Common 'Solve Options'
#
_solve_options_group_title = 'Solve Options'

safe_declare_unique_option(
    common_block,
    "mipgap",
    PySPConfigValue(
        None,
        domain=_domain_unit_interval,
        description=(
            "Specifies the mipgap for all sub-problems (scenarios or bundles). "
            "The default value of None indicates not mipgap should be used."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver_options",
    PySPConfigValue(
        (),
        domain=_domain_tuple_of_str_or_dict,
        description=(
            "Persistent solver options for all sub-problems (scenarios or bundles). "
            "This option can be used multiple times from the command line to specify "
            "more than one solver option."
        ),
        doc=None,
        visibility=0),
    ap_kwds={'action': 'append'},
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver",
    PySPConfigValue(
        "cplex",
        domain=_domain_must_be_str,
        description=(
            "Specify the solver with which to solve scenarios "
            "(or bundles). Default is cplex."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver_io",
    PySPConfigValue(
        None,
        domain=_domain_must_be_str,
        description=(
            "The type of IO used to execute the solver. Different "
            "solvers support different types of IO, but the following "
            "are common options: lp - generate LP files, nl - generate "
            "NL files, python - direct Python interface, os - generate "
            "OSiL XML files."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver_manager",
    PySPConfigValue(
        "serial",
        domain=_domain_must_be_str,
        description=(
            "The type of solver manager used to coordinate scenario "
            "sub-problem solves. Default is serial."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "disable_warmstart",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Disable warm-start of all sub-problem solves."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver_manager_pyro_host",
    PySPConfigValue(
        None,
        domain=_domain_must_be_str,
        description=(
            "The hostname to bind on when searching for a Pyro "
            "nameserver controlling the solver manager."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver_manager_pyro_port",
    PySPConfigValue(
        None,
        domain=int,
        description=(
            "The port to bind on when searching for a Pyro "
            "nameserver controlling the solver_manager."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

safe_declare_unique_option(
    common_block,
    "solver_manager_pyro_shutdown",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Attempt to shut down all Pyro-related components "
            "associated with the Pyro name server for the "
            "solver manager. Components to shutdown include "
            "the name server, dispatch server, and any "
            "pyro_mip_server processes. Note that if Pyro4 is "
            "in use the nameserver will always ignore this "
            "request."
        ),
        doc=None,
        visibility=0),
    ap_group=_solve_options_group_title)

#
# Common 'Other Options'
#
_other_options_group_title = 'Other Options'

safe_declare_unique_option(
    common_block,
    "disable_gc",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Disable the python garbage collecter."
        ),
        doc=None,
        visibility=0),
    ap_group=_other_options_group_title)

safe_declare_unique_option(
    common_block,
    "profile_memory",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "If Guppy or Pympler is available, report memory usage statistics "
            "for objects created by various PySP constructs. The default value "
            "of 0 disables memory profiling. Values greater than 0 indiciate "
            "increasing levels of verbosity."
        ),
        doc=None,
        visibility=0),
    ap_group=_other_options_group_title)

safe_declare_unique_option(
    common_block,
    "profile",
    PySPConfigValue(
        0,
        domain=_domain_nonnegative_integer,
        description=(
            "Enable profiling of Python code. The value of this "
            "option is the number of functions that are summarized. "
            "The default value of 0 disables profiling."
        ),
        doc=None,
        visibility=0),
    ap_group=_other_options_group_title)

safe_declare_unique_option(
    common_block,
    "traceback",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "When an exception is thrown, show the entire call "
            "stack. Ignored if profiling is enabled."
        ),
        doc=None,
        visibility=0),
    ap_group=_other_options_group_title)

safe_declare_unique_option(
    common_block,
    "compile_scenario_instances",
    PySPConfigValue(
        False,
        domain=bool,
        description=(
            "Replace all linear constraints on scenario instances with "
            "a more memory efficient sparse matrix representation."
        ),
        doc=None,
        visibility=0),
    ap_group=_other_options_group_title)

#
# Deprecated command-line option names
# (DO NOT REGISTER THEM OUTSIDE OF THIS FILE)
#
_map_to_deprecated = {}
_deprecated_block = \
    PySPConfigBlock("A collection of common deprecated PySP command-line options")
_deprecated_options_group_title = "Deprecated Options"

#
# --model-directory
#
class _DeprecatedModelDirectory(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedModelDirectory, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--model-directory' command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use --model-location instead.")
        setattr(namespace, 'CONFIGBLOCK.model_location', values)

def _warn_model_directory(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'model_directory' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'model_location'. "
        "Please use 'model_location' instead.\n")
    return _domain_must_be_str(val)

safe_declare_unique_option(
    _deprecated_block,
    "model_directory",
    PySPConfigValue(
        None,
        domain=_warn_model_directory,
        description=(
            "Deprecated alias for --model-location"
        ),
        doc=None,
        visibility=1),
    ap_args=("--model-directory",),
    ap_kwds={'action':_DeprecatedModelDirectory},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['model_location'] = \
    _deprecated_block.get('model_directory')

#
# -i, --instance-directory
#
class _DeprecatedInstanceDirectory(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedInstanceDirectory, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
#            logger.warning(
#                "DEPRECATED: The '--instance-directory' ('-i') command-line "
#                "option has been deprecated and will be removed "
#                "in the future. Please use '--scenario-tree-location' ('-s') instead.")
        setattr(namespace, 'CONFIGBLOCK.scenario_tree_location', values)

def _warn_instance_directory(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'instance_directory' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'scenario_tree_location'. "
        "Please use 'scenario_tree_location' instead.\n")
    return _domain_must_be_str(val)

safe_declare_unique_option(
    _deprecated_block,
    "instance_directory",
    PySPConfigValue(
        None,
        domain=_warn_instance_directory,
        description=(
            "Deprecated alias for --scenario-tree-location, -s"
        ),
        doc=None,
        visibility=1),
    ap_args=("-i", "--instance-directory"),
    ap_kwds={'action':_DeprecatedInstanceDirectory},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['scenario_tree_location'] = \
    _deprecated_block.get('instance_directory')

#
# --handshake-with-phpyro
#

class _DeprecatedHandshakeWithPHPyro(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedHandshakeWithPHPyro, self).\
            __init__(option_strings, dest, nargs=0, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--handshake-with-phpyro command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--pyro-handshake-at-startup instead.")
        setattr(namespace, 'CONFIGBLOCK.pyro_handshake_at_startup', True)

def _warn_handshake_with_phpyro(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'handshake_with_phpyro' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'pyro_handshake_at_startup'. "
        "Please use 'pyro_handshake_at_startup' instead.\n")
    return bool(val)

safe_declare_unique_option(
    _deprecated_block,
    "handshake_with_phpyro",
    PySPConfigValue(
        None,
        domain=_warn_handshake_with_phpyro,
        description=(
            "Deprecated alias for --pyro-handshake-at-startup"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedHandshakeWithPHPyro},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['pyro_handshake_at_startup'] = \
    _deprecated_block.get('handshake_with_phpyro')

#
# --phpyro-required-workers
#

class _DeprecatedPHPyroRequiredWorkers(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedPHPyroRequiredWorkers, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--phpyro-required-workers command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--pyro-required-scenariotreeservers instead.")
        setattr(namespace, 'CONFIGBLOCK.pyro_required_scenariotreeservers', values)

def _warn_phpyro_required_workers(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'phpyro_required_workers' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'pyro_required_scenariotreeservers'. "
        "Please use 'pyro_required_scenariotreeservers' instead.\n")
    return _domain_nonnegative_integer(val)

safe_declare_unique_option(
    _deprecated_block,
    "phpyro_required_workers",
    PySPConfigValue(
        None,
        domain=_warn_phpyro_required_workers,
        description=(
            "Deprecated alias for --pyro-required-scenariotreeservers"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedPHPyroRequiredWorkers},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['pyro_required_scenariotreeservers'] = \
    _deprecated_block.get('phpyro_required_workers')

#
# --phpyro-workers-timeout
#

class _DeprecatedPHPyroWorkersTimeout(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedPHPyroWorkersTimeout, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--phpyro-workers-timeout command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--pyro-find-scenariotreeservers-timeout instead.")
        setattr(namespace, 'CONFIGBLOCK.pyro_find_scenariotreeservers_timeout', values)

def _warn_phpyro_workers_timeout(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'phpyro_workers_timeout' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'pyro_find_scenariotreeservers_timeout'. "
        "Please use 'pyro_find_scenariotreeservers_timeout' instead.\n")
    return float(val)

safe_declare_unique_option(
    _deprecated_block,
    "phpyro_workers_timeout",
    PySPConfigValue(
        None,
        domain=_warn_phpyro_workers_timeout,
        description=(
            "Deprecated alias for --pyro-find-scenariotreeservers-timeout"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedPHPyroWorkersTimeout},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['pyro_find_scenariotreeservers_timeout'] = \
    _deprecated_block.get('phpyro_workers_timeout')

#
# --phpyro-transmit-leaf-stage-variable-solutions
#

class _DeprecatedPHPyroTransmitLeafStageVariableSolutions(
        argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedPHPyroTransmitLeafStageVariableSolutions, self).\
            __init__(option_strings, dest, nargs=0, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--phpyro-transmit-leaf-stage-variable-solutions "
            "command-line option has been deprecated and will be removed "
            "in the future. Please use "
            "'--pyro-transmit-leaf-stage-variable-solutions instead.")
        setattr(namespace,
                'CONFIGBLOCK.pyro_transmit_leaf_stage_variable_solutions',
                True)

def _warn_phpyro_transmit_leaf_stage_variable_solutions(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'phpyro_transmit_leaf_stage_variable_solutions' config "
        "item will be ignored unless it is being used as a command-line option "
        "where it can be redirected to "
        "'pyro_transmit_leaf_stage_variable_solutions'. Please use "
        "'pyro_transmit_leaf_stage_variable_solutions' instead.\n")
    return bool(val)

safe_declare_unique_option(
    _deprecated_block,
    "phpyro_transmit_leaf_stage_variable_solutions",
    PySPConfigValue(
        None,
        domain=_warn_phpyro_transmit_leaf_stage_variable_solutions,
        description=(
            "Deprecated alias for --pyro-transmit-leaf-stage-variable-solutions"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedPHPyroTransmitLeafStageVariableSolutions},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['pyro_transmit_leaf_stage_variable_solutions'] = \
    _deprecated_block.get('phpyro_transmit_leaf_stage_variable_solutions')

#
# --scenario-tree-seed
#

class _DeprecatedScenarioTreeSeed(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedScenarioTreeSeed, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--scenario-tree-seed command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--scenario-tree-random-seed instead.")
        setattr(namespace, 'CONFIGBLOCK.scenario_tree_random_seed', values)

def _warn_scenario_tree_seed(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'scenario_tree_seed' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'scenario_tree_random_seed'. "
        "Please use 'scenario_tree_random_seed' instead.\n")
    return int(val)

safe_declare_unique_option(
    _deprecated_block,
    "scenario_tree_seed",
    PySPConfigValue(
        None,
        domain=_warn_scenario_tree_seed,
        description=(
            "Deprecated alias for --scenario-tree-random-seed"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedScenarioTreeSeed},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['scenario_tree_random_seed'] = \
    _deprecated_block.get('scenario_tree_seed')

#
# --scenario-mipgap
#

class _DeprecatedScenarioMipGap(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedScenarioMipGap, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--scenario-mipgap command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--mipgap instead.")
        setattr(namespace, 'CONFIGBLOCK.mipgap', values)

def _warn_scenario_mipgap(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'scenario_mipgap' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'mipgap'. "
        "Please use 'mipgap' instead.\n")
    return _domain_unit_interval(val)

safe_declare_unique_option(
    _deprecated_block,
    "scenario_mipgap",
    PySPConfigValue(
        None,
        domain=_warn_scenario_mipgap,
        description=(
            "Deprecated alias for --mipgap"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedScenarioMipGap},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['mipgap'] = \
    _deprecated_block.get('scenario_mipgap')

#
# --scenario-solver-options
#

class _DeprecatedScenarioSolverOptions(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedScenarioSolverOptions, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--scenario-solver-options command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--solver-options instead.")
        current = getattr(namespace, 'CONFIGBLOCK.solver_options', values)
        current.append(values)

#
# --bounds-cfgfile
#

class _DeprecatedBoundsCFGFile(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedBoundsCFGFile, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--bounds-cfgfile command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--postinit-callback-location instead.")
        current = getattr(namespace,
                          'CONFIGBLOCK.postinit_callback_location')
        current.append(values)

def _warn_bounds_cfgfile(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'bounds_cfgfile' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'postinit_callback_location'. "
        "Please use 'postinit_callback_location' instead.\n")
    return _domain_tuple_of_str(val)

safe_declare_unique_option(
    _deprecated_block,
    "bounds_cfgfile",
    PySPConfigValue(
        None,
        domain=_warn_bounds_cfgfile,
        description=(
            "Deprecated alias for --postinit-callback-location"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedBoundsCFGFile},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['postinit_callback_location'] = \
    _deprecated_block.get('bounds_cfgfile')

#
# --aggregate-cfgfile
#

class _DeprecatedAggregateCFGFile(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedAggregateCFGFile, self).\
            __init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--aggregate-cfgfile command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--aggregategetter-callback-location "
            "instead.")
        current = getattr(namespace,
                          'CONFIGBLOCK.aggregategetter_callback_location')
        current.append(values)

def _warn_aggregate_cfgfile(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'aggregate_cfgfile' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'aggregategetter_callback_location'. "
        "Please use 'aggregategetter_callback_location' instead.\n")
    return _domain_tuple_of_str(val)

safe_declare_unique_option(
    _deprecated_block,
    "aggregate_cfgfile",
    PySPConfigValue(
        None,
        domain=_warn_aggregate_cfgfile,
        description=(
            "Deprecated alias for --aggregategetter-callback-location"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedAggregateCFGFile},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['aggregategetter_callback_location'] = \
    _deprecated_block.get('aggregate_cfgfile')

#
# --shutdown-pyro
#

class _DeprecatedShutdownPyro(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedShutdownPyro, self).\
            __init__(option_strings, dest, nargs=0, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--shutdown-pyro command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--pyro-shutdown "
            "instead.")
        setattr(namespace,
                'CONFIGBLOCK.pyro_shutdown',
                True)

def _warn_shutdown_pyro(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'shutdown_pyro' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'pyro_shutdown'. "
        "Please use 'pyro_shutdown' instead.\n")
    return bool(val)

safe_declare_unique_option(
    _deprecated_block,
    "shutdown_pyro",
    PySPConfigValue(
        None,
        domain=_warn_shutdown_pyro,
        description=(
            "Deprecated alias for --pyro-shutdown"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedShutdownPyro},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['pyro_shutdown'] = \
    _deprecated_block.get('shutdown_pyro')

class _DeprecatedShutdownPyroWorkers(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(_DeprecatedShutdownPyroWorkers, self).\
            __init__(option_strings, dest, nargs=0, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning(
            "DEPRECATED: The '--shutdown-pyro-workers command-line "
            "option has been deprecated and will be removed "
            "in the future. Please use '--pyro-shutdown-workers "
            "instead.")
        setattr(namespace,
                'CONFIGBLOCK.pyro_shutdown_workers',
                True)

def _warn_shutdown_pyro_workers(val):
    # don't use logger here since users might not import
    # the pyomo logger in a scripting interface
    sys.stderr.write(
        "\tWARNING: The 'shutdown_pyro_workers' config item will be ignored "
        "unless it is being used as a command-line option "
        "where it can be redirected to 'pyro_shutdown_workers'. "
        "Please use 'pyro_shutdown_workers' instead.\n")
    return bool(val)

safe_declare_unique_option(
    _deprecated_block,
    "shutdown_pyro_workers",
    PySPConfigValue(
        None,
        domain=_warn_shutdown_pyro_workers,
        description=(
            "Deprecated alias for --pyro-shutdown-workers"
        ),
        doc=None,
        visibility=1),
    ap_kwds={'action':_DeprecatedShutdownPyroWorkers},
    ap_group=_deprecated_options_group_title,
    declare_for_argparse=True)
_map_to_deprecated['pyro_shutdown_workers'] = \
    _deprecated_block.get('shutdown_pyro_workers')

#
# Register a common option
#
def safe_declare_common_option(configblock,
                               name,
                               prefix=None,
                               ap_group=None):
    assert isinstance(configblock, PySPConfigBlock)
    assert name not in _deprecated_block
    assert name in common_block
    common_value = common_block.get(name)
    assert common_value._parent == common_block
    assert common_value._userSet == False
    assert common_value._userAccessed == False
    if prefix is not None:
        if common_value._argparse is not None:
            raise ValueError(
                "Cannot register a common option with a prefix "
                "when the PySPConfigValue has already been declared "
                "with argparse data"
                "short name")
        if name in _map_to_deprecated:
            raise ValueError(
                "Cannot register a common option with a prefix "
                "when the common option is mapped to a deprecated "
                "option name")
        name = prefix + name
        if name in _map_to_deprecated:
            raise ValueError(
                "Cannot register a common option with a prefix "
                "when the prefixed name is mapped to a deprecated "
                "option name")
    if name not in configblock:
        common_value._parent = None
        common_value_copy = copy.deepcopy(common_value)
        common_value._parent = common_block
        if ap_group is not None:
            common_value_copy.ap_group = ap_group
        configblock.declare(name, common_value_copy)
        #
        # handle deprecated command-line option names
        #
        if name in _map_to_deprecated:
            deprecated_value = _map_to_deprecated[name]
            assert deprecated_value._parent == _deprecated_block
            assert deprecated_value._userSet == False
            assert deprecated_value._userAccessed == False
            deprecated_value._parent = None
            deprecated_value_copy = copy.deepcopy(deprecated_value)
            deprecated_value._parent = _deprecated_block
            configblock.declare(deprecated_value_copy._name, deprecated_value_copy)
    else:
        current = configblock.get(name)
        check_options_match(current, common_value)
        if ap_group is not None:
            current.ap_group = ap_group
        if name in _map_to_deprecated:
            deprecated_value = _map_to_deprecated[name]
            assert deprecated_value._name in configblock
            current = configblock.get(deprecated_value._name)
            check_options_match(current, deprecated_value)

#
# Register a common option and make sure it is declared for argparse
#
def safe_register_common_option(configblock,
                                name,
                                prefix=None):

    assert isinstance(configblock, PySPConfigBlock)
    assert name not in _deprecated_block
    assert name in common_block, name
    common_value = common_block.get(name)
    assert common_value._parent == common_block
    assert common_value._userSet == False
    assert common_value._userAccessed == False
    if prefix is not None:
        if common_value._argparse is not None:
            raise ValueError(
                "Cannot register a common option with a prefix "
                "when the PySPConfigValue has already been declared "
                "with argparse data"
                "short name")
        if name in _map_to_deprecated:
            raise ValueError(
                "Cannot register a common option with a prefix "
                "when the common option is mapped to a deprecated "
                "option name")
        name = prefix + name
        if name in _map_to_deprecated:
            raise ValueError(
                "Cannot register a common option with a prefix "
                "when the prefixed name is mapped to a deprecated "
                "option name")
    if name not in configblock:
        common_value._parent = None
        common_value_copy = copy.deepcopy(common_value)
        common_value._parent = common_block
        configblock.declare(name, common_value_copy)
        if common_value_copy._argparse is None:
            _kwds = {}
            if common_value.ap_group is not None:
                _kwds['group'] = common_value.ap_group
            common_value_copy.declare_as_argument(**_kwds)
        #
        # handle deprecated command-line option names
        #
        if name in _map_to_deprecated:
            deprecated_value = _map_to_deprecated[name]
            assert deprecated_value._parent == _deprecated_block
            assert deprecated_value._userSet == False
            assert deprecated_value._userAccessed == False
            deprecated_value._parent = None
            deprecated_value_copy = copy.deepcopy(deprecated_value)
            deprecated_value._parent = _deprecated_block
            configblock.declare(deprecated_value_copy._name, deprecated_value_copy)
    else:
        current = configblock.get(name)
        if common_value._argparse is not None:
            check_options_match(current,
                                common_value)
        else:
            check_options_match(current,
                                common_value,
                                include_argparse=False)
        if name in _map_to_deprecated:
            deprecated_value = _map_to_deprecated[name]
            if deprecated_value._name not in configblock:
                assert deprecated_value._parent == _deprecated_block
                assert deprecated_value._userSet == False
                assert deprecated_value._userAccessed == False
                deprecated_value._parent = None
                deprecated_value_copy = copy.deepcopy(deprecated_value)
                deprecated_value._parent = _deprecated_block
                configblock.declare(deprecated_value_copy._name,
                                    deprecated_value_copy)
            current = configblock.get(deprecated_value._name)
            check_options_match(current, deprecated_value)

if __name__ == "__main__":
    import pyomo.environ

    class Junk1(object):

        @staticmethod
        def register_options(config_block):
            common_option_names = [
                'model_location',
                'scenario_tree_location',
                'objective_sense_stage_based']
            for name in common_option_names:
                safe_declare_common_option(config_block, name)
                if config_block.get(name)._argparse is None:
                    config_block.get(name).declare_as_argument()

    class Junk2(object):
        @staticmethod
        def register_options(config_block):
            for name in common_block:
                safe_declare_common_option(config_block, name)
                if config_block.get(name)._argparse is None:
                    config_block.get(name).declare_as_argument()


    block = PySPConfigBlock()
    #Junk1.register_options(block)
    Junk2.register_options(block)
    block.declare('b', PySPConfigBlock())
    #Junk3.register_options(block.b)

    ap = argparse.ArgumentParser()
    block.initialize_argparse(ap)
    block.import_argparse(ap.parse_args())

    #print block.generate_yaml_template()
    #print block.model_location
    #block.model_location = 'gabe'
    #print block.model_location
    #print block['model_location']
    #print block.get('model_location')
    #print list(block.user_values())
    #print list(block.unused_user_values())
    #block.model_location = '2'
    #block.phpyro_transmit_leaf_stage_variable_solutions = 1
    #print 'model_location' in block
    #print 'model location' in block
#    block.solution_writer
    #print type(block.model_location)
    print("")
    print(block.bounds_cfgfile)
    print(block.postinit_callback_location)
    print(list((_c._name, _c.value(False)) for _c in block.user_values()))
    print(list(_c._name for _c in block.unused_user_values()))

    #options = PySPConfigBlock()
    #options.model_location = common.get('model_location')

    print("Deprecation Pairs:")
    for name in _map_to_deprecated:
        print(_map_to_deprecated[name]._name+" -> "+name)
