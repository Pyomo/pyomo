#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('PySPConfiguredObject', 'PySPConfiguredExtension')

import inspect
import copy
import sys
from collections import defaultdict

from pyomo.util.plugin import SingletonPlugin
from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyomo.pysp.util.config import (safe_register_option,
                                    check_options_match,
                                    safe_register_common_option)

import six

#
# All classes that derive from PySPConfiguredObject should declare a
# _registered_options ConfigBlock at the class level and populate it
# with ConfigValue objects. This base class will handle the rest. To
# avoid name conflicts when declared with a prefix, all
# implementations should check for option values using
# self.get_option(<name>) rather than directly accessing the _options
# member.
#
# Note: This class is designed to allow appearing multiple
#       times in an inheritance hierarchy. All derived classes
#       should call their base class __init__ using super()
#

class PySPConfiguredObject(object):

    _registered_options = \
        ConfigBlock("Options registered for the PySPConfiguredObject class")

    def __init__(self, options, prefix=""):
        self._prefix = None
        self._options = None
        self.set_prefix(prefix)
        if options is not None:
            self.set_options(options)

    def get_option(self, name):
        assert self._options is not None
        configval = self._options.get(self._prefix + name)
        return configval.value()

    def set_prefix(self, prefix):
        if not isinstance(prefix, six.string_types):
            raise TypeError(
            "Prefix value must be a built-in "
            "string type, not '%s'" % (type(prefix)))
        self._prefix = prefix

    def set_options(self, options):
        """Assign an options block to this class instance after
        validating that all registered options for this class exist on
        the block."""
        self.validate_options(options, prefix=self._prefix)
        self._options = options

    def display_options(self, ostream=sys.stdout):
        """Display the options block currently assigned to this class
        instance."""
        if (self._options is None) or \
           (len(self._options) == 0):
            ostream.write("None\n")
        else:
            max_len = max(len(name) for name in self._options)
            line_template = " %s %"+str(max_len)+"s: %s\n"
            for name in self._options:
                configval = self._options.get(name)
                flag = "-"
                if configval._userSet:
                    flag = "*"
                ostream.write(line_template
                              % (flag, name,configval.value(accessValue=False)))

    @classmethod
    def register_options(cls, *args, **kwds):
        """Cls.register_options([options]) -> options.  Fills an
        options block will all registered options for this class. The
        optional argument 'options' can be a previously existing
        options block, which would be both updated and returned by
        this function. The optional flag 'prefix' can be set to
        indicate that all class options should be registered with the
        given prefix prepended to their original name."""
        prefix = kwds.pop('prefix',"")
        assert isinstance(prefix, six.string_types)
        if len(kwds) != 0:
            raise ValueError("Unexpected keywords: %s"
                             % (str(kwds)))
        if len(args) == 0:
            options = ConfigBlock()
        else:
            if len(args) != 1:
                raise TypeError(
                    "register_options(...) takes at most 1 argument (%s given)"
                    % (len(args)))
            options = args[0]
            if not isinstance(options, ConfigBlock):
                raise TypeError(
                    "register_options(...) argument must be of type ConfigBlock, "
                    "not %s" % (type(options).__name__))

        bases = inspect.getmro(cls)
        assert bases[-1] is object
        #
        # Register options in order of most derived class first.
        # This allows derived classes to update option default values.
        # The safe_register_option function will simply validate that
        # an existing option matches when a base class attempts to register
        # an option that is already registered (in which cases we tell it
        # to accept differences in default values).
        #
        for base in bases:
            #
            # The check here is that PySPConfiguredObject needs to
            # appear as an immediate base in a class definition for us
            # to check the list of _registered options declared in
            # that class definitions immediate scope. This allows
            # _registered_options to be distributed across different
            # containers in the class hierarchy, while leaving the
            # ability for derived classes to NOT have to define an
            # empty _registered_options block if they don't have any
            # new options to add but are derived from some class which
            # does use PySPConfiguredObject as a base class. By not
            # declaring PySPConfiguredObject as an immediate base
            # class, we know not to check for a _registered_options
            # data member in this derived class's scope (because
            # otherwise we'd be getting some base class's definition)
            #
            if any(base is PySPConfiguredObject for base in base.__bases__):
                for name in base._registered_options:
                    configval = base._registered_options.get(name)
                    assert configval._parent is base._registered_options
                    configval._parent = None
                    declare_for_argparse = False
                    if (configval._argparse is None) and \
                       (options.get(prefix+name, None) is None):
                        declare_for_argparse = True
                    safe_register_option(options,
                                         prefix+name,
                                         configval,
                                         relax_default_check=True,
                                         declare_for_argparse=declare_for_argparse)
                    configval._parent = base._registered_options
        return options

    @classmethod
    def extract_options(cls,
                        options,
                        prefix="",
                        error_if_missing=True):
        """Copy the set of registered options for this class from an
        existing options block and return a new options block with
        only those values. This method will preserve the _userSet
        status of all options. The optional flag 'prefix' can be set
        to indicate that all registered class options will have a name
        predended with the given prefix. The optional flag
        'error_if_missing' controls whether or not an exception is
        raised when registered options are missing."""

        assert isinstance(prefix, six.string_types)
        these_options = cls.register_options(prefix=prefix)
        for name in these_options.keys():
            configval = None
            try:
                configval = options.get(name)
            except KeyError:
                configval = None
                if error_if_missing:
                    raise
                else:
                    continue
            assert configval is not None
            this_configval = these_options.get(name)
            check_options_match(this_configval,
                                configval,
                                include_default=False,
                                include_value=False,
                                include_accessed=False)
            if configval._userSet:
                this_configval.set_value(configval.value())

        return these_options

    @classmethod
    def extract_user_options_to_dict(cls,
                                     options,
                                     prefix="",
                                     error_if_missing=True,
                                     sparse=False):
        """Copy the set of registered options for this class from an
        existing options block and return a dictionary of options
        (name -> value) with those values. This method will preserve
        the _userSet status of all options. The optional flag 'prefix'
        can be set to indicate that all registered class options will
        have a name predended with the given prefix. The optional flag
        'error_if_missing' controls whether or not an exception is
        raised when registered options are missing. The optional flag
        'sparse' controls whether non user-set values should be
        included in the returned dictionary."""

        assert isinstance(prefix, six.string_types)
        these_options = \
            cls.extract_options(options,
                                prefix=prefix,
                                error_if_missing=error_if_missing)
        options_dict = {}
        for name in these_options:
            configval = these_options.get(name)
            if (not sparse) or configval._userSet:
                options_dict[name] = configval.value()
        return options_dict

    @classmethod
    def validate_options(cls,
                         options,
                         prefix="",
                         error_if_missing=True):
        """Validate that all registered options can be found in the
        options block and that their option definitions are the
        same. The optional flag 'prefix' can be set to indicate that
        all registered class options will have a name predended with
        the given prefix. The optional flag 'error_if_missing' can be
        used to control whether or not an exception is raised when
        registered options are missing."""

        assert isinstance(prefix, six.string_types)
        bases = inspect.getmro(cls)
        first = defaultdict(lambda: True)
        for base in bases:
            # The check here is that PySPConfiguredObject needs to
            # appear as an immediate base in a class definition for us
            # to check the list of _registered options declared in
            # that class definitions immediate scope. This allows
            # _registered_options to be distributed across different
            # containers in the class hierarchy, while leaving the
            # ability for derived classes to NOT have to define an
            # empty _registered_options block if they don't have any
            # new options to add but are derived from some class which
            # does use PySPConfiguredObject as a base class. By not
            # declaring PySPConfiguredObject as an immediate base
            # class, we know not to check for a _registered_options
            # data member in this derived class's scope (because
            # otherwise we'd be getting some base class's definition)
            #
            if any(base is PySPConfiguredObject for base in base.__bases__):
                for name in base._registered_options:
                    configval = None
                    try:
                        configval = options.get(prefix + name)
                    except KeyError:
                        configval = None
                        if error_if_missing:
                            raise
                        else:
                            continue
                    assert configval is not None
                    this_configval = base._registered_options.get(name)
                    include_argparse = False
                    if this_configval._argparse is not None:
                        include_argparse = True
                    check_options_match(this_configval,
                                        configval,
                                        include_argparse=include_argparse,
                                        include_default=first[prefix + name],
                                        include_value=False,
                                        include_accessed=False)
                    first[prefix + name] = False

#
# Classes that derive from PySPConfiguredExtension are meant to be
# implemented as SingletonPlugin objects. By default, they prefix all
# registered option names by their class name, but the implementor can
# override this by declaring _default_prefix at the class level in
# their implementation.
#
# Note: This class is NOT designed appear multiple times
#       in an inheritance hierarchy. All derived classes
#       must directly call this bases class __init__ method
#       rather than relying on super(). This is due to
#       how Pyutilib implements its Singleton type, causing
#       the __class__ cell to be empty.
#       (See: https://stackoverflow.com/questions/
#             13126727/how-is-super-in-python-3-implemented)
#

class PySPConfiguredExtension(PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the PySPConfiguredExtension class")

    safe_register_common_option(_registered_options,
                                "extension_precedence")

    _default_prefix = None

    def __init__(self):
        PySPConfiguredObject.__init__(self, None, prefix=self.extension_prefix())
        if not isinstance(self, SingletonPlugin):
            raise TypeError("PySPConfiguredExtension implementations must "
                            "inherit from SingletonPlugin")

    @classmethod
    def extension_prefix(cls):
        bases = inspect.getmro(cls)
        assert bases[-1] is object
        for base in bases:
            if getattr(base, '_default_prefix', None) is not None:
                return base._default_prefix
        return cls.__name__.lower()+"_"

    @classmethod
    def register_options(cls, *args, **kwds):
        """Cls.register_options([options]) -> options.  Fills an
        options block will all registered options for this class. The
        optional argument 'options' can be a previously existing
        options block, which would be both updated and returned by
        this function."""
        if 'prefix' not in kwds:
            kwds['prefix'] = cls.extension_prefix()
        return super(PySPConfiguredExtension, cls).register_options(*args, **kwds)

    @classmethod
    def extract_options(cls,
                        options,
                        prefix=None,
                        error_if_missing=True):
        """Copy the set of registered options for this class from an
        existing options block and return a new options block with
        only those values. This method will preserve the _userSet
        status of all options. The optional flag 'prefix' can be set
        to indicate that all registered class options will have a name
        predended with the given prefix. The optional flag
        'error_if_missing' controls whether or not an exception is
        raised when registered options are missing."""

        if prefix is None:
            prefix = cls.extension_prefix()
        return super(PySPConfiguredExtension, cls).extract_options(
            options,
            prefix=prefix,
            error_if_missing=error_if_missing)

    @classmethod
    def extract_user_options_to_dict(cls,
                                     options,
                                     prefix=None,
                                     error_if_missing=True,
                                     sparse=False):
        """Copy the set of registered options for this class from an
        existing options block and return a dictionary of options
        (name -> value) with those values. This method will preserve
        the _userSet status of all options. The optional flag 'prefix'
        can be set to indicate that all registered class options will
        have a name predended with the given prefix. The optional flag
        'error_if_missing' controls whether or not an exception is
        raised when registered options are missing. The optional flag
        'sparse' controls whether non user-set values should be
        included in the returned dictionary."""

        if prefix is None:
            prefix = cls.extension_prefix()
        return super(PySPConfiguredExtension, cls).extract_user_options_to_dict(
            options,
            prefix=prefix,
            error_if_missing=error_if_missing,
            sparse=sparse)

    @classmethod
    def validate_options(cls,
                         options,
                         prefix=None,
                         error_if_missing=True):
        """Validate that all registered options can be found in the
        options block and that their option definitions are the
        same. The optional flag 'prefix' can be set to indicate that
        all registered class options will have a name predended with
        the given prefix. The optional flag 'error_if_missing' can be
        used to control whether or not an exception is raised when
        registered options are missing."""

        if prefix is None:
            prefix = cls.extension_prefix()
        return super(PySPConfiguredExtension, cls).validate_options(
            options,
            prefix=prefix,
            error_if_missing=error_if_missing)
