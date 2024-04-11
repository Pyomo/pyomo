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

from pyomo.contrib.cspline_external.cspline_parameters import (
    cubic_parameters_model,
    get_parameters,
)
import pyomo.environ as pyo


if __name__ == "__main__":

    x_data = [
        1,
        2,
        3,
        4,
        5,
    ]

    y_data = [
        2,
        3,
        5,
        2,
        1,
    ]

    m = cubic_parameters_model(
        x_data,
        y_data,
        end_point_constraint=True,
        objective_form=False,
    )

    # REMOVE, temporary just to set solver path
    import idaes

    # solver_obj = pyo.SolverFactory("clp")
    solver_obj = pyo.SolverFactory("ipopt")
    solver_obj.solve(m, tee=True)

    get_parameters(m, "test.txt")
