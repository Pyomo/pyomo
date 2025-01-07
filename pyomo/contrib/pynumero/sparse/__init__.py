#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import numpy_available, scipy_available

try:
    if numpy_available and scipy_available:
        from .block_vector import BlockVector, NotFullyDefinedBlockVectorError
        from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
except ImportError as e:
    print("IMPORT ERROR: ", e)
    print("Current environment information...")
    import sys
    import platform
    import pkg_resources

    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.platform()})")

    print("\nInstalled packages:")
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [f"{pkg.key}=={pkg.version}" for pkg in installed_packages]
    )
    print("\n".join(installed_packages_list))

    print("\nImported packages:")
    imported_packages = sorted(sys.modules.keys())
    print("\n".join(imported_packages))
    raise e
