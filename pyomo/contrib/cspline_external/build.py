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

import sys
from pyomo.common.cmake_builder import build_cmake_project


def build_cspline_external(user_args=[], parallel=None):
    return build_cmake_project(
        targets=["src"],
        package_name="cspline_1d_external",
        description="ASL external cubic spline interpolation function library",
        user_args=["-DBUILD_AMPLASL_IF_NEEDED=ON"] + user_args,
        parallel=parallel,
    )


class ASLCsplineExternalBuilder(object):
    def __call__(self, parallel):
        return build_cspline_external(parallel=parallel)


if __name__ == "__main__":
    build_cspline_external(sys.argv[1:])
