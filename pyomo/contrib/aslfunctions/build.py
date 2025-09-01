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


def build_aslfunctions(user_args=[], parallel=None):
    return build_cmake_project(
        targets=["src"],
        package_name="aslfunctions",
        description="Useful AMPL external functions",
        user_args=["-DBUILD_AMPLASL_IF_NEEDED=ON"] + user_args,
        parallel=parallel,
    )


class LibASLFunctionsBuilder(object):
    def __call__(self, parallel):
        return build_aslfunctions(parallel=parallel)


if __name__ == "__main__":
    build_aslfunctions(sys.argv[1:])
