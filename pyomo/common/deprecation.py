"""Decorator for deprecating functions."""

import logging
import functools
import inspect
import textwrap

from pyomo.common.errors import DeveloperError

def deprecation_warning(msg, logger='pyomo.core'):
    """Standardized formatter for deprecation warnings

    This is a standardized routine for formatting deprecation warnings
    so that things look consistent and "nice".

    Args:
        msg (str): the deprecation message to format
    """
    logging.getLogger(logger).warning(
        textwrap.fill('DEPRECATED: %s' % (msg,), width=70) )

def deprecated( msg=None, logger='pyomo.core', version=None, remove_in=None ):
    """Indicate that a function, method or class is deprecated.

    This decorator will cause a warning to be logged when the wrapped
    function or method is called, or when the deprecated class is
    constructed.  This decorator also updates the target object's
    docstring to indicate that it is deprecated.

    Args:
        msg (str): a custom deprecation message (default: "This
            {function|class} has been deprecated and may be
            removed in a future release.")

        logger (str): the logger to use for emitting the warning
            (default: "pyomo.core")

        version (str): [required] the version in which the decorated
            object was deprecated.  General practice is to set version
            to '' or 'TBD' during development and update it to the
            actual release as part of the release process.

        remove_in (str): the version in which the decorated object will be
            removed from the code.

    """
    if version is None:  # or version in ('','tbd','TBD'):
        raise DeveloperError("@deprecated missing initial version")

    def wrap(func):
        message = _default_msg(msg, func)

        @functools.wraps(func, assigned=('__module__', '__name__'))
        def wrapper(*args, **kwargs):
            deprecation_warning(message, logger)
            return func(*args, **kwargs)

        if func.__doc__ is None:
            wrapper.__doc__ = textwrap.fill(
                'DEPRECATION WARNING: %s' % (message,), width=70)
        else:
            wrapper.__doc__ = textwrap.fill(
                'DEPRECATION WARNING: %s' % (message,), width=70) + '\n\n' + \
                textwrap.fill(textwrap.dedent(func.__doc__.strip()))
        return wrapper

    def _default_msg(user_msg, func):
        if user_msg is None:
            if inspect.isclass(func):
                _obj = ' class'
            elif inspect.isfunction(func):
                _obj = ' function'
            else:
                _obj = ''
            user_msg = 'This%s has been deprecated and may be removed in a ' \
                       'future release.' % (_obj,)
        comment = []
        if version:
            comment.append('deprecated in %s' % (version,))
        if remove_in:
            comment.append('will be removed in %s' % (remove_in))
        if comment:
            user_msg += "  (%s)" % (','.join(comment))
        return user_msg

    return wrap
