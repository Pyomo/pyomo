#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# backport of contextlib.nullcontext for supporting Python < 3.7
class nullcontext(object):
    def __init__(self, enter_result=None):
        self.result = enter_result

    def __enter__(self):
        return self.result

    def __exit__(self, et, ev, tb):
        return
