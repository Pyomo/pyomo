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

import pyomo.environ as pyo
from ReactorDesign import create_model

# set the data (native Python data)
k1 = 5.0 / 6.0  # min^-1
k2 = 5.0 / 3.0  # min^-1
k3 = 1.0 / 6000.0  # m^3/(gmol min)

# solve the model for different values of caf and report results
print('{:>10s}\t{:>10s}\t{:>10s}'.format('CAf', 'SV', 'CB'))
for cafi in range(1, 11):
    caf = cafi * 1000.0  # gmol/m^3

    # create the model with the new data
    # note, we could do this more efficiently with
    # mutable parameters
    m = create_model(k1, k2, k3, caf)

    # solve the problem
    status = pyo.SolverFactory('ipopt').solve(m)
    print("{:10g}\t{:10g}\t{:10g}".format(caf, pyo.value(m.sv), pyo.value(m.cb)))
