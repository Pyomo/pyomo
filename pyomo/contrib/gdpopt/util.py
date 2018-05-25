"""Utility functions and classes for the GDPopt solver."""
from __future__ import division


class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class GDPoptSolveData(object):
    """Data container to hold solve-instance data."""
    pass
