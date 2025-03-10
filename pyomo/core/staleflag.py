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


class _StaleFlagManager(object):
    def __init__(self):
        self._current = 0
        self.mark_all_as_stale()

    def stale_mapper(self, encode, value):
        if encode:
            return self.is_stale(value)
        else:
            if value:
                return 0
            else:
                return self.get_flag(0)

    def _get_flag(self, current_flag):
        """Return the current global stale flag value"""
        return self._current

    def _get_flag_delayed(self, current_flag):
        """Implement the "delayed" advancement of the global stale flag value

        This will continue to return the current value of the state flag
        until the first non-stale variable is updated (that it, it is
        passed the current stale flag when called).  This allows for
        updating stale variable values without incrementing the global
        stale flag, but will mark everything as stale as soon as a
        non-stale variable value is changed.

        """
        if current_flag == self._current:
            self._current += 1
            setattr(self, 'get_flag', getattr(self, '_get_flag'))
        return self._current

    def is_stale(self, val):
        """Return ``True`` if the passed value indicated a stale variable"""
        return val != self._current

    def mark_all_as_stale(self, delayed=False):
        """Advance the global stale flag, marking all variables as stale

        This is generally called immediately before and after a batch
        variable update (i.e. loading values from a solver result or
        stored solution).  Before the batch update
        :meth:`mark_all_as_stale` is called with ``delayed=False``,
        which immediately marks all variables as stale.  After the batch
        update, :meth:`mark_all_as_stale` is typically called with
        ``delayed=True``.  This allows additional stale variables to be
        updated without advancing the global flag, but as soon as any
        non-stale variable has its value changed, then the flag is
        advanced and all other variables become stale.

        """
        if delayed:
            setattr(self, 'get_flag', getattr(self, '_get_flag_delayed'))
        else:
            setattr(self, 'get_flag', getattr(self, '_get_flag'))
            self._current += 1


StaleFlagManager = _StaleFlagManager()
