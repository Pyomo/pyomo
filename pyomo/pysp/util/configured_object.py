#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('PySPConfiguredObject', 'PySPConfiguredExtension')

import inspect
import copy
import sys
from collections import defaultdict

from pyomo.util.plugin import SingletonPlugin
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_declare_option,
                                    check_options_match,
                                    safe_declare_common_option,
                                    _deprecated_block)

import six

class PySPConfiguredObject(object):
    """
    A base class for classes that declare options using
    on a PySPConfigBlock.

    All classes that derive from PySPConfiguredObject should
    defined a _declare_options classmethod that returns or
    updates a PySPConfigBlock populated with with
    PySPConfigValue objects. This base class will handle the
    rest. To avoid name conflicts when declared with a
    prefix, all implementations should check for option
    values using self.get_option(<name>) rather than
    directly accessing the _options member.

    This class is designed to allow appearing multiple times
    in an inheritance hierarchy. All derived classes should
    call their base class __init__ using super().
    """

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()
        return options

    def __init__(self, options, options_prefix=""):
        self._options_prefix = None
        self._options = None
        self.set_options_prefix(options_prefix)
        if options is not None:
            self.set_options(options)

    def get_option(self, name):
        """Get the value for the option with the input name. The
        options prefix currently set will automatically be prepended to the
        name."""
        assert self._options is not None
        configval = self._options.get(self._options_prefix + name)
        return configval.value()

    def get_full_option_name(self, name):
        """Return the full option name obtained by prepending the
        current options prefix to the input name."""
        assert self._options is not None
        configval = self._options.get(self._options_prefix + name)
        return self._options_prefix + name

    def set_options_prefix(self, options_prefix):
        """Set the options prefix used to read options from the current
        options object assigned to this class."""
        if not isinstance(options_prefix, six.string_types):
            raise TypeError(
            "Options prefix must be a built-in "
            "string type, not '%s'" % (type(options_prefix)))
        self._options_prefix = options_prefix

    def set_options(self, options):
        """Assign an options block to this class instance after
        validating that all registered options for this class exist
        on the block."""
        self.validate_options(options, options_prefix=self._options_prefix)
        self._options = options

    def display_options(self,
                        ostream=None,
                        registered_only=False):
        """Display the options on the options block currently
        assigned to this class instance. Options that have been
        explicitly set will be marked with a '*' symbol. The optional
        argument 'registered_only' can be set to indicate that only
        options registered to this class should be displayed."""
        if (self._options is None) or \
           (len(self._options) == 0):
            # Note: writing to directly to stdout rather
            #       than using 'print' as the default causes
            #       issues with some of the redirection used
            #       in test_ph.py
            if ostream is None:
                print("None")
            else:
                ostream.write("None\n")
        else:
            if registered_only:
                names = \
                    list(cls.register_options(options_prefix=self._options_prefix).keys())
            else:
                names = list(self._options.keys())
            max_len = max(len(name) for name in names)
            line_template = " %s %"+str(max_len)+"s: %s"
            for name in names:
                configval = self._options.get(name)
                flag = "-"
                if configval._userSet:
                    flag = "*"
                out = (line_template
                       % (flag,name,configval.value(accessValue=False)))
                if name in _deprecated_block:
                    out += " (DEPRECATED)"
                if ostream is None:
                    print(out)
                else:
                    ostream.write(out+"\n")

    @classmethod
    def register_options(cls, *args, **kwds):
        """Cls.register_options([options]) -> options. Fills an
        options block will all registered options for this
        class. The optional argument 'options' can be a previously
        existing options block, which would be both updated and
        returned by this function. The optional flag 'options_prefix' can be
        set to indicate that all class options should be registered
        with the given prefix prepended to their original name."""

        options_prefix = kwds.pop('options_prefix',"")
        assert isinstance(options_prefix, six.string_types)
        setup_argparse = kwds.pop('setup_argparse',True)
        if len(kwds) != 0:
            raise ValueError("Unexpected keywords: %s"
                             % (str(kwds)))
        if len(args) == 0:
            options = PySPConfigBlock()
        else:
            if len(args) != 1:
                raise TypeError(
                    "register_options(...) takes at most 1 argument (%s given)"
                    % (len(args)))
            options = args[0]
            if not isinstance(options, PySPConfigBlock):
                raise TypeError(
                    "register_options(...) argument must be of "
                    "type PySPConfigBlock, not %s"
                    % (type(options).__name__))

        bases = inspect.getmro(cls)
        assert bases[-1] is object
        #
        # Register options in order of most derived class
        # first. This allows derived classes to update
        # option default values. The safe_declare_option
        # function will simply validate that an existing
        # option matches when a base class attempts to
        # register an option that is already registered (in
        # which cases we tell it to accept differences in
        # default values).
        #
        for base in bases:
            #
            # The check here is that PySPConfiguredObject
            # needs to appear as an immediate base in a
            # class definition for us to assume that class
            # is declaring options. This allows declared
            # options to be distributed across different
            # containers in the class hierarchy, while
            # leaving the ability for derived classes to NOT
            # have to define an empty _declared_options
            # block or dummy _declare_options classmethod if
            # they don't have any new options to add but are
            # derived from some class which does use
            # PySPConfiguredObject as a base class. By not
            # declaring PySPConfiguredObject as an immediate
            # base class, we know not to check for a
            # _declared_options or _declare options in this
            # derived class's scope (because otherwise we'd
            # be getting some base class's definition)
            #
            if any(_base is PySPConfiguredObject
                   for _base in base.__bases__):
                # make sure we've removed this older form
                assert not hasattr(base, "_declared_options")
                if "_declare_options" in base.__dict__:
                    _declared_options = base._declare_options()
                    for name in _declared_options:
                        configval = _declared_options.get(name)
                        assert configval._parent is _declared_options
                        configval._parent = None
                        declare_for_argparse = False
                        if (setup_argparse) and \
                           (configval._argparse is None) and \
                           (options_prefix+name not in options):
                            declare_for_argparse = True
                        safe_declare_option(
                            options,
                            options_prefix+name,
                            configval,
                            relax_default_check=True,
                            declare_for_argparse=declare_for_argparse)
                        configval._parent = _declared_options
        return options

    @classmethod
    def extract_options(cls,
                        options,
                        options_prefix="",
                        source_options_prefix="",
                        error_if_missing=True):
        """Copy the set of registered options for this class from an
        existing options block and return a new options block with
        only those values. This method will preserve the _userSet
        status of all options. The optional flag 'options_prefix' can be set
        to indicate that all registered class options in the
        returned options object will have a name prepended with the
        given prefix. The optional flag 'source_options_prefix' can be set to
        indicate that all registered class options on the input
        options object have a named prepended with the given
        prefix. The optional flag 'error_if_missing' controls
        whether or not an exception is raised when registered
        options are missing from the input options object."""

        assert isinstance(options_prefix, six.string_types)
        bare_options = cls.register_options(options_prefix="")
        return_options = cls.register_options(options_prefix=options_prefix)
        for name in bare_options.keys():
            configval = None
            if source_options_prefix+name in options:
                configval = options.get(source_options_prefix+name)
            elif error_if_missing:
                raise KeyError(source_options_prefix+name)
            else:
                continue
            assert configval is not None
            this_configval = return_options.get(options_prefix+name)
            include_argparse = False
            check_options_match(this_configval,
                                configval,
                                include_default=True,
                                include_argparse=include_argparse,
                                include_value=False,
                                include_accessed=False)
            if configval._userSet:
                this_configval.set_value(configval.value())

        return return_options

    @classmethod
    def extract_user_options_to_dict(cls,
                                     options,
                                     options_prefix="",
                                     source_options_prefix="",
                                     error_if_missing=True,
                                     sparse=False):
        """Copy the set of registered options for this class from an
        existing options block and return a dictionary of options
        (name -> value) with those values. This method will preserve
        the _userSet status of all options. The optional flag
        'options_prefix' can be set to indicate that all registered class
        options will have a name prepended with the given prefix in
        the output dictionary. The optional flag 'source_options_prefix' can be
        set to indicate that all registered class options on the
        input options object have a named prepended with the given
        prefix. The optional flag 'error_if_missing' controls
        whether or not an exception is raised when registered
        options are missing from the input options object. The
        optional flag 'sparse' controls whether non user-set values
        should be included in the returned dictionary."""

        assert isinstance(options_prefix, six.string_types)
        bare_options = \
            cls.extract_options(options,
                                options_prefix="",
                                source_options_prefix=source_options_prefix,
                                error_if_missing=error_if_missing)
        options_dict = {}
        for name in bare_options:
            configval = bare_options.get(name)
            if (not sparse) or configval._userSet:
                options_dict[options_prefix+name] = configval.value()
        return options_dict

    @classmethod
    def update_options_from_argparse(cls,
                                     options,
                                     ap_data,
                                     options_prefix="",
                                     source_options_prefix="",
                                     skip_userset=False,
                                     error_if_missing=True):
        """Update the input options object by extracting all registered
        options for this class from an argparse Namespace object. This
        method cannot determine if the values on the argparse Namespace
        object were set explicitly or are defaults. Therefore, the
        _userSet status will be updated on all options that are found.
        The method only compares the names against the attributes found
        on the argparse Namespace object. No other form of validation
        is performed. The optional flag 'options_prefix' can be set to indicate
        that all registered class options will have a name prepended
        with the given prefix on the updated options object. The
        optional flag 'source_options_prefix' can be set to indicate that
        registered class option names should be prepended with the
        the given prefix when searching for items on the argparse
        Namespace object. The optional flag 'error_if_missing' controls
        whether or not an exception is raised when registered option
        names are missing from the argparse Namespace object."""

        assert isinstance(options_prefix, six.string_types)
        assert isinstance(source_options_prefix, six.string_types)
        bare_options = \
            cls.extract_options(options,
                                options_prefix="",
                                error_if_missing=error_if_missing)
        for name in bare_options:
            configval = options.get(options_prefix+name)
            if (not skip_userset) or (not configval._userSet):
                try:
                    ap_val = getattr(ap_data, source_options_prefix+name)
                except AttributeError:
                    if error_if_missing:
                        raise
                else:
                    configval.set_value(ap_val)

    @classmethod
    def validate_options(cls,
                         options,
                         options_prefix="",
                         error_if_missing=True):
        """Validate that all registered options can be found in the
        options block and that their option definitions are the
        same. The optional flag 'options_prefix' can be set to indicate that
        all registered class options will have a name prepended with
        the given prefix. The optional flag 'error_if_missing' can
        be used to control whether or not an exception is raised
        when registered options are missing."""

        assert isinstance(options_prefix, six.string_types)
        bases = inspect.getmro(cls)
        first = defaultdict(lambda: True)
        for base in bases:
            # The check here is that PySPConfiguredObject needs to
            # appear as an immediate base in a class definition for us
            # to check the list of _registered options declared in
            # that class definitions immediate scope. This allows
            # _declared_options to be distributed across different
            # containers in the class hierarchy, while leaving the
            # ability for derived classes to NOT have to define an
            # empty _declared_options block if they don't have any
            # new options to add but are derived from some class which
            # does use PySPConfiguredObject as a base class. By not
            # declaring PySPConfiguredObject as an immediate base
            # class, we know not to check for a _declared_options
            # data member in this derived class's scope (because
            # otherwise we'd be getting some base class's definition)
            #
            if any(base is PySPConfiguredObject
                   for base in base.__bases__):
                # make sure we've removed this older form
                assert not hasattr(base, "_declared_options")
                if "_declare_options" in base.__dict__:
                    _declared_options = base._declare_options()
                    for name in _declared_options:
                        configval = None
                        if options_prefix+name in options:
                            configval = options.get(options_prefix+name)
                        elif error_if_missing:
                            raise KeyError(options_prefix+name)
                        else:
                            continue
                        assert configval is not None
                        this_configval = _declared_options.get(name)
                        include_argparse = False
                        if this_configval._argparse is not None:
                            include_argparse = True
                        check_options_match(this_configval,
                                            configval,
                                            include_argparse=include_argparse,
                                            include_default=first[options_prefix + name],
                                            include_value=False,
                                            include_accessed=False)
                        first[options_prefix + name] = False

#
# Classes that derive from PySPConfiguredExtension are meant to be
# implemented as SingletonPlugin objects. By default, they prefix all
# registered option names by their class name, but the implementor can
# override this by declaring _default_options_prefix at the class level in
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

    @classmethod
    def _declare_options(cls, options=None):
        if options is None:
            options = PySPConfigBlock()

        safe_declare_common_option(options,
                                   "extension_precedence")

        return options

    _default_options_prefix = None

    def __init__(self):
        PySPConfiguredObject.__init__(self,
                                      None,
                                      options_prefix=self.extension_options_prefix())
        if not isinstance(self, SingletonPlugin):
            raise TypeError("PySPConfiguredExtension implementations must "
                            "inherit from SingletonPlugin")

    @classmethod
    def extension_options_prefix(cls):
        bases = inspect.getmro(cls)
        assert bases[-1] is object
        for base in bases:
            if getattr(base, '_default_options_prefix', None) is not None:
                return base._default_options_prefix
        return cls.__name__.lower()+"_"

    @classmethod
    def register_options(cls, *args, **kwds):
        """Cls.register_options([options]) -> options. Fills an
        options block will all registered options for this
        class. The optional argument 'options' can be a previously
        existing options block, which would be both updated and
        returned by this function. The optional flag 'options_prefix' can be
        set to indicate that all class options should be registered
        with the given prefix prepended to their original name."""

        if 'options_prefix' not in kwds:
            kwds['options_prefix'] = cls.extension_options_prefix()
        return super(PySPConfiguredExtension, cls).\
            register_options(*args, **kwds)

    @classmethod
    def extract_options(cls,
                        options,
                        options_prefix=None,
                        source_options_prefix=None,
                        error_if_missing=True):
        """Copy the set of registered options for this class from an
        existing options block and return a new options block with
        only those values. This method will preserve the _userSet
        status of all options. The optional flag 'options_prefix' can be set
        to indicate that all registered class options in the
        returned options object will have a name prepended with the
        given prefix. The optional flag 'source_options_prefix' can be set to
        indicate that all registered class options on the input
        options object have a named prepended with the given
        prefix. The optional flag 'error_if_missing' controls
        whether or not an exception is raised when registered
        options are missing from the input options object."""

        if options_prefix is None:
            options_prefix = cls.extension_options_prefix()
        if source_options_prefix is None:
            source_options_prefix = cls.extension_options_prefix()
        return super(PySPConfiguredExtension, cls).extract_options(
            options,
            options_prefix=options_prefix,
            source_options_prefix=source_options_prefix,
            error_if_missing=error_if_missing)

    @classmethod
    def extract_user_options_to_dict(cls,
                                     options,
                                     options_prefix=None,
                                     source_options_prefix=None,
                                     error_if_missing=True,
                                     sparse=False):
        """Copy the set of registered options for this class from an
        existing options block and return a dictionary of options
        (name -> value) with those values. This method will preserve
        the _userSet status of all options. The optional flag
        'options_prefix' can be set to indicate that all registered class
        options will have a name prepended with the given prefix in
        the output dictionary. The optional flag 'source_options_prefix' can be
        set to indicate that all registered class options on the
        input options object have a named prepended with the given
        prefix. The optional flag 'error_if_missing' controls
        whether or not an exception is raised when registered
        options are missing from the input options object. The
        optional flag 'sparse' controls whether non user-set values
        should be included in the returned dictionary."""

        if options_prefix is None:
            options_prefix = cls.extension_options_prefix()
        if source_options_prefix is None:
            source_options_prefix = cls.extension_options_prefix()
        return super(PySPConfiguredExtension, cls).\
            extract_user_options_to_dict(
                options,
                options_prefix=options_prefix,
                source_options_prefix=source_options_prefix,
                error_if_missing=error_if_missing,
                sparse=sparse)

    @classmethod
    def update_options_from_argparse(cls,
                                     options,
                                     ap_data,
                                     options_prefix=None,
                                     source_options_prefix=None,
                                     skip_userset=False,
                                     error_if_missing=True):
        """Update the input options object by extracting all
        registered options for this class from an argparse Namespace
        object. This method cannot determine if the values on the
        argparse Namespace object were set explicitly or are
        defaults. Therefore, the _userSet status will be updated on
        all options that are found.  The method only compares the
        names against the attributes found on the argparse Namespace
        object. No other form of validation is performed. The
        optional flag 'options_prefix' can be set to indicate that all
        registered class options will have a name prepended with the
        given prefix on the updated options object. The optional
        flag 'source_options_prefix' can be set to indicate that registered
        class option names should be prepended with the the given
        prefix when searching for items on the argparse Namespace
        object. The optional flag 'skip_userset' can be set to
        indicate that options with the _userSet flag already set to
        True (on the options object being updated) should be skipped
        when loading options from the argparse Namespace object. The
        optional flag 'error_if_missing' controls whether or not an
        exception is raised when registered option names are missing
        from the argparse Namespace object."""

        if options_prefix is None:
            options_prefix = cls.extension_options_prefix()
        if source_options_prefix is None:
            source_options_prefix = cls.extension_options_prefix()
        return super(PySPConfiguredExtension, cls).\
            update_options_from_argparse(
                options,
                ap_data,
                options_prefix=options_prefix,
                source_options_prefix=source_options_prefix,
                skip_userset=skip_userset,
                error_if_missing=error_if_missing)

    @classmethod
    def validate_options(cls,
                         options,
                         options_prefix=None,
                         error_if_missing=True):
        """Validate that all registered options can be found in the
        options block and that their option definitions are the
        same. The optional flag 'options_prefix' can be set to indicate that
        all registered class options will have a name prepended with
        the given prefix. The optional flag 'error_if_missing' can
        be used to control whether or not an exception is raised
        when registered options are missing."""

        if options_prefix is None:
            options_prefix = cls.extension_options_prefix()
        return super(PySPConfiguredExtension, cls).validate_options(
            options,
            options_prefix=options_prefix,
            error_if_missing=error_if_missing)
