#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.opt import WriterFactory


def write_conflict_set(m, filename):
    """
    For debugging infeasible CPs: writes the conflict set found by CP optimizer
    to a file with the specified filename.

    Args:
        m: Pyomo CP model
        filename: string filename
    """

    cpx_mod, var_map = WriterFactory('docplex_model').write(
        m, symbolic_solver_labels=True
    )
    conflict = cpx_mod.refine_conflict()
    conflict.write(filename)
