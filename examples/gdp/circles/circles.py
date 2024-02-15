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

"""
The "circles" GDP example problem originating in Lee and Grossman (2000). The 
goal is to choose a point to minimize a convex quadratic function over a set of 
disjoint hyperspheres.
"""

from pyomo.environ import (
    ConcreteModel,
    Objective,
    Param,
    RangeSet,
    Var,
    Constraint,
    value,
)
import matplotlib as mpl
import matplotlib.pyplot as plt


# Circles2D3 is the original example described in the papers. Ruiz and Grossman
# (2012) discuss two more example instances, Circles2D36 and Circles3D36, but I
# (S. Davis 08/2023) couldn't find data for them, so I made a couple.
# format: dimension, circle_centers, circle_rvals, circle_penalties, reference_point, upper_bound
circles_model_examples = {
    "Circles2D3": {
        'dimension': 2,
        # written as {(circle number, coordinate index): value}, for example below
        # means {circle 1: (0, 0), circle 2: (4, 1), circle 3: (2, 4)},
        'circle_centers': {
            (1, 1): 0,
            (1, 2): 0,
            (2, 1): 4,
            (2, 2): 1,
            (3, 1): 2,
            (3, 2): 4,
        },
        'circle_rvals': {1: 1, 2: 1, 3: 1},
        'circle_penalties': {1: 2, 2: 1, 3: 3},
        'reference_point': {1: 3, 2: 2},
        'upper_bound': 8,
    },
    # Here the solver has a local minimum to avoid by not choosing the closest point
    "Circles2D3_modified": {
        'dimension': 2,
        'circle_centers': {
            (1, 1): 0,
            (1, 2): 0,
            (2, 1): 4,
            (2, 2): 1,
            (3, 1): 2,
            (3, 2): 4,
        },
        'circle_rvals': {1: 1, 2: 1, 3: 1},
        'circle_penalties': {1: 2, 2: 3, 3: 1},
        'reference_point': {1: 3, 2: 2},
        'upper_bound': 8,
    },
    # Here's a 3D model.
    "Circles3D4": {
        'dimension': 3,
        'circle_centers': {
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (2, 1): 2,
            (2, 2): 0,
            (2, 3): 1,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 3,
            (4, 1): 1,
            (4, 2): 6,
            (4, 3): 1,
        },
        'circle_rvals': {1: 1, 2: 1, 3: 1, 4: 1},
        'circle_penalties': {1: 1, 2: 3, 3: 2, 4: 1},
        'reference_point': {1: 2, 2: 2, 3: 1},
        'upper_bound': 10,
    },
}


def build_model(data=circles_model_examples["Circles2D3"]):
    """Build a model. By default you get Circles2D3, a small instance."""

    # Ensure good data was passed
    assert len(data) == 6, "Error processing data"

    dimension = data['dimension']
    circle_centers = data['circle_centers']
    circle_rvals = data['circle_rvals']
    circle_penalties = data['circle_penalties']
    reference_point = data['reference_point']
    upper_bound = data['upper_bound']

    assert len(circle_rvals) == len(circle_penalties), "Error processing data"
    assert len(circle_centers) == len(circle_rvals) * dimension, "Error processing data"
    assert len(reference_point) == dimension, "Error processing data"

    m = ConcreteModel()

    m.circles = RangeSet(len(circle_rvals), doc=f"{len(circle_rvals)} circles")
    m.idx = RangeSet(dimension, doc="n-dimensional indexing set for coordinates")

    m.circ_centers = Param(
        m.circles, m.idx, initialize=circle_centers, doc="Center points of the circles"
    )
    m.circ_rvals = Param(
        m.circles, initialize=circle_rvals, doc="Squared radii of circles"
    )
    m.ref_point = Param(
        m.idx,
        initialize=reference_point,
        doc="Reference point for distance calculations",
    )
    m.circ_penalties = Param(
        m.circles, initialize=circle_penalties, doc="Penalty for being in each circle"
    )

    # Choose a point to minimize objective
    m.point = Var(
        m.idx,
        bounds=(0, upper_bound),
        doc="Chosen point at which we evaluate objective function",
    )

    # Let's set up the "active penalty" this way
    m.active_penalty = Var(
        bounds=(0, max(m.circ_penalties[i] for i in m.circles)),
        doc="Penalty for being in the current circle",
    )

    # Disjunction: we must be in at least one circle (in fact exactly one since they are disjoint)
    @m.Disjunct(m.circles)
    def circle_disjunct(d, circ):
        m = d.model()
        d.inside_circle = Constraint(
            expr=sum((m.point[i] - m.circ_centers[circ, i]) ** 2 for i in m.idx)
            <= m.circ_rvals[circ]
        )
        d.penalty = Constraint(expr=m.active_penalty == m.circ_penalties[circ])

    @m.Disjunction()
    def circle_disjunction(m):
        return [m.circle_disjunct[i] for i in m.circles]

    # Objective function is Euclidean distance from reference point, plus a penalty depending on which circle we are in
    m.cost = Objective(
        expr=sum((m.point[i] - m.ref_point[i]) ** 2 for i in m.idx) + m.active_penalty,
        doc="Distance from reference point plus penalty",
    )

    return m


def draw_model(m, title=None):
    """Draw a model using matplotlib to illustrate what's going on. Pass 'title' arg to give chart a title"""

    if len(m.idx) != 2:
        print(
            f"Unable to draw: this drawing code only supports 2D models, but a {len(m.idx)}-dimensional model was passed."
        )
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    if title is not None:
        plt.title(title)

    for c in m.circles:
        x = m.circ_centers[c, 1]
        y = m.circ_centers[c, 2]
        # remember, we are keeping around the *squared* radii for some reason
        r = m.circ_rvals[c] ** 0.5
        penalty = m.circ_penalties[c]
        print(f"drawing circle {c}: x={x}, y={y}, r={r}, penalty={penalty}")

        ax.add_patch(
            mpl.patches.Circle(
                (x, y), radius=r, facecolor="#0d98e6", edgecolor="#000000"
            )
        )
        # the alignment appears to work by magic
        ax.text(
            x,
            y + r + 0.3,
            f"Circle {c}: penalty {penalty}",
            horizontalalignment="center",
        )

    # now the reference point
    ax.add_patch(
        mpl.patches.Circle(
            (m.ref_point[1], m.ref_point[2]),
            radius=0.05,
            facecolor="#902222",
            edgecolor="#902222",
        )
    )
    ax.text(
        m.ref_point[1],
        m.ref_point[2] + 0.15,
        "Reference Point",
        horizontalalignment="center",
    )

    # and the chosen point
    ax.add_patch(
        mpl.patches.Circle(
            (value(m.point[1]), value(m.point[2])),
            radius=0.05,
            facecolor="#11BB11",
            edgecolor="#11BB11",
        )
    )
    ax.text(
        value(m.point[1]),
        value(m.point[2]) + 0.15,
        "Chosen Point",
        horizontalalignment="center",
    )

    plt.show()


if __name__ == "__main__":
    from pyomo.environ import SolverFactory
    from pyomo.core.base import TransformationFactory

    # Set up a solver, for example scip and bigm works
    solver = SolverFactory("scip")
    transformer = TransformationFactory("gdp.bigm")

    for key in circles_model_examples:
        model = build_model(circles_model_examples[key])
        print(f"Solving model {key}")
        transformer.apply_to(model)
        solver.solve(model)
        pt_string = (
            "("
            + "".join(
                str(value(model.point[i])) + (", " if i != len(model.idx) else "")
                for i in model.idx
            )
            + ")"
        )
        print(f"Optimal value found: {model.cost()} with chosen point {pt_string}")
        draw_model(model, title=key)
        print()
