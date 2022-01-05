#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

class _StaleFlagManager(object):
    def __init__(self):
        self._current = 1
        self._next = 1
        self.advance_flag()

    def _get_flag(self):
        return self._current

    def _get_flag_first(self):
        self._current = self._next
        setattr(self, 'get_flag', getattr(self, '_get_flag'))
        return self._current

    def is_stale(self, val):
        return val != self._current

    def advance_flag(self):
        setattr(self, 'get_flag', getattr(self, '_get_flag_first'))
        self._current = self._next
        self._next += 1

StaleFlagManager = _StaleFlagManager()
