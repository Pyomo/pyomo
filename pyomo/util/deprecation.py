"""Decorator for deprecating functions."""

__author__ = "Qi Chen <https://github.com/qtothec>"

import logging
import functools
import textwrap


def deprecated(msg='This function has been deprecated and may be removed '
               'in a future release',
               logger='pyomo.core'):
    """Indicate that a function is deprecated.

    This decorator will cause a warning to be logged when the wrapped function
    is called.

    """
    def wrap(func):
        @functools.wraps(func, assigned=('__module__', '__name__'))
        def wrapper(*args, **kwargs):
            logging.getLogger(logger).warning('DEPRECATED: {0}'.format(msg))
            return func(*args, **kwargs)
        if func.__doc__ is None:
            wrapper.__doc__ = textwrap.fill(
                'DEPRECATION WARNING: {0}'.format(msg), width=70)
        else:
            wrapper.__doc__ = textwrap.fill(
                'DEPRECATION WARNING: {0}'.format(msg), width=70) + '\n\n' + \
                textwrap.fill(textwrap.dedent(func.__doc__.strip()))
        return wrapper
    return wrap
