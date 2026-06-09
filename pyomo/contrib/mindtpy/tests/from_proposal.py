# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""MINLP test model adapted from material in David Bernal's PhD thesis proposal.

References
----------
Bernal, D. E. Convex MINLP project update derived from the thesis-proposal
material.
ResearchGate project page: https://www.researchgate.net/project/Convex-MINLP/update/5c7eb2ee3843b034242e9e4a
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Reals,
    Integers,
    Objective,
    Var,
    sqrt,
    minimize,
)
from pyomo.common.collections import ComponentMap


class FromProposalModel(ConcreteModel):
    """MINLP benchmark adapted from David Bernal's PhD thesis proposal material."""

    def __init__(self, *args, **kwargs):
        """Create the problem.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``ConcreteModel``.
        **kwargs
            Keyword arguments forwarded to ``ConcreteModel``.
        """
        kwargs.setdefault('name', 'DavidProposalExample')
        super(FromProposalModel, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(domain=Reals, bounds=(0, 20), initialize=1)
        m.y = Var(domain=Integers, bounds=(0, 20), initialize=4)

        m.c1 = Constraint(expr=m.x**2 / 20.0 + m.y <= 20)
        m.c2 = Constraint(expr=(m.x - 1) ** 2 / 40.0 - m.y <= -4)
        m.c3 = Constraint(expr=m.y - 10 * sqrt(m.x + 0.1) <= 0)
        m.c4 = Constraint(expr=-m.x - m.y <= -5)

        m.objective = Objective(expr=m.x - m.y / 4.5 + 2, sense=minimize)
        m.optimal_value = 0.66555
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 1.1099999999999999
        m.optimal_solution[m.y] = 11.0
