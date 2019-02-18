
try:
    from .ip_solver import InteriorPointSolver
except ImportError:
    print("Need MA27 or MUMPS to run pynumero interior-point. "
          "conda install -c conda-forge pymumps")

try:
    from .cyipopt_solver import CyIpoptSolver
except ImportError:
    print("Need CyIpopt to run CyipoptSolver"
          "conda install -c conda-forge cyipopt")