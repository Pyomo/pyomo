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

"""2-D constrained layout example.

Example based on: https://www.minlp.org/library/problem/index.php?i=107&lib=GDP
Description can be found in: https://doi.org/10.1016/j.ejor.2011.10.002

This model attempts to pack a set of rectangles within a set of circles while
minimizing the cost of connecting the rectangles, a function of the distance
between the rectangle centers. It is assumed that the circles do not overlap
with each other.

"""

from pyomo.environ import ConcreteModel, Objective, Param, RangeSet, Set, Var, value

# Constrained layout model examples. These are from Nicolas Sawaya (2006).
# Format: rect_lengths, rect_heights, circ_xvals, circ_yvals, circ_rvals (as dicts),
# sep_penalty_matrix (as nested array)
# Note that only the strict upper triangle of sep_penalty_matrix is used
constrained_layout_model_examples = {
    "CLay0203": {
        'rect_lengths': {1: 5, 2: 7, 3: 3},
        'rect_heights': {1: 6, 2: 5, 3: 3},
        'circ_xvals': {1: 15, 2: 50},
        'circ_yvals': {1: 10, 2: 80},
        'circ_rvals': {1: 6, 2: 5},
        'sep_penalty_matrix': [[0, 300, 240], [0, 0, 100]],
    },
    "CLay0204": {
        'rect_lengths': {1: 5, 2: 7, 3: 3, 4: 2},
        'rect_heights': {1: 6, 2: 5, 3: 3, 4: 3},
        'circ_xvals': {1: 15, 2: 50},
        'circ_yvals': {1: 10, 2: 80},
        'circ_rvals': {1: 6, 2: 10},
        'sep_penalty_matrix': [[0, 300, 240, 210], [0, 0, 100, 150], [0, 0, 0, 120]],
    },
    "CLay0205": {
        'rect_lengths': {1: 5, 2: 7, 3: 3, 4: 2, 5: 9},
        'rect_heights': {1: 6, 2: 5, 3: 3, 4: 3, 5: 7},
        'circ_xvals': {1: 15, 2: 50},
        'circ_yvals': {1: 10, 2: 80},
        'circ_rvals': {1: 6, 2: 10},
        'sep_penalty_matrix': [
            [0, 300, 240, 210, 50],
            [0, 0, 100, 150, 30],
            [0, 0, 0, 120, 25],
            [0, 0, 0, 0, 60],
        ],
    },
    "CLay0303": {
        'rect_lengths': {1: 5, 2: 7, 3: 3},
        'rect_heights': {1: 6, 2: 5, 3: 3},
        'circ_xvals': {1: 15, 2: 50, 3: 30},
        'circ_yvals': {1: 10, 2: 80, 3: 50},
        'circ_rvals': {1: 6, 2: 5, 3: 4},
        'sep_penalty_matrix': [[0, 300, 240], [0, 0, 100]],
    },
    "CLay0304": {
        'rect_lengths': {1: 5, 2: 7, 3: 3, 4: 2},
        'rect_heights': {1: 6, 2: 5, 3: 3, 4: 3},
        'circ_xvals': {1: 15, 2: 50, 3: 30},
        'circ_yvals': {1: 10, 2: 80, 3: 50},
        'circ_rvals': {1: 6, 2: 5, 3: 4},
        'sep_penalty_matrix': [[0, 300, 240, 210], [0, 0, 100, 150], [0, 0, 0, 120]],
    },
    "CLay0305": {
        'rect_lengths': {1: 5, 2: 7, 3: 3, 4: 2, 5: 9},
        'rect_heights': {1: 6, 2: 5, 3: 3, 4: 3, 5: 7},
        'circ_xvals': {1: 15, 2: 50, 3: 30},
        'circ_yvals': {1: 10, 2: 80, 3: 50},
        'circ_rvals': {1: 6, 2: 10, 3: 4},
        'sep_penalty_matrix': [
            [0, 300, 240, 210, 50],
            [0, 0, 100, 150, 30],
            [0, 0, 0, 120, 25],
            [0, 0, 0, 0, 60],
        ],
    },
}


def build_constrained_layout_model(
    params=constrained_layout_model_examples['CLay0203'], metric="l1"
):
    """Build the model."""

    # Ensure the caller passed good data
    assert len(params) == 6

    # Get all the parameters out
    rect_lengths = params['rect_lengths']
    rect_heights = params['rect_heights']
    circ_xvals = params['circ_xvals']
    circ_yvals = params['circ_yvals']
    circ_rvals = params['circ_rvals']
    sep_penalty_matrix = params['sep_penalty_matrix']

    assert len(rect_lengths) == len(
        rect_heights
    ), "There should be the same number of rectangle lengths and heights."
    assert (
        len(circ_xvals) == len(circ_yvals) == len(circ_rvals)
    ), "There should be the same number of circle x values, y values, and radii"
    for row in sep_penalty_matrix:
        assert len(row) == len(
            sep_penalty_matrix[0]
        ), "Matrix rows should have the same length"
    assert metric in ["l1", "l2"], 'Metric options are "l1" and "l2"'

    m = ConcreteModel(name="2-D constrained layout")
    m.rectangles = RangeSet(len(rect_lengths), doc=f"{len(rect_lengths)} rectangles")
    m.circles = RangeSet(len(circ_xvals), doc=f"{len(circ_xvals)} circles")

    m.rect_length = Param(m.rectangles, initialize=rect_lengths, doc="Rectangle length")
    m.rect_height = Param(m.rectangles, initialize=rect_heights, doc="Rectangle height")

    m.circle_x = Param(
        m.circles, initialize=circ_xvals, doc="x-coordinate of circle center"
    )
    m.circle_y = Param(
        m.circles, initialize=circ_yvals, doc="y-coordinate of circle center"
    )
    m.circle_r = Param(m.circles, initialize=circ_rvals, doc="radius of circle")

    @m.Param(m.rectangles, doc="Minimum feasible x value for rectangle")
    def x_min(m, rect):
        return min(
            m.circle_x[circ] - m.circle_r[circ] + m.rect_length[rect] / 2
            for circ in m.circles
        )

    @m.Param(m.rectangles, doc="Maximum feasible x value for rectangle")
    def x_max(m, rect):
        return max(
            m.circle_x[circ] + m.circle_r[circ] - m.rect_length[rect] / 2
            for circ in m.circles
        )

    @m.Param(m.rectangles, doc="Minimum feasible y value for rectangle")
    def y_min(m, rect):
        return min(
            m.circle_y[circ] - m.circle_r[circ] + m.rect_height[rect] / 2
            for circ in m.circles
        )

    @m.Param(m.rectangles, doc="Maximum feasible y value for rectangle")
    def y_max(m, rect):
        return max(
            m.circle_y[circ] + m.circle_r[circ] - m.rect_height[rect] / 2
            for circ in m.circles
        )

    m.rect_pairs = Set(
        initialize=m.rectangles * m.rectangles, filter=lambda _, r1, r2: r1 < r2
    )

    m.rect_sep_penalty = Param(
        m.rect_pairs,
        # 0-based vs 1-based indices...
        initialize={
            (r1, r2): sep_penalty_matrix[r1 - 1][r2 - 1] for r1, r2 in m.rect_pairs
        },
        doc="Penalty for separation distance between two rectangles.",
    )

    def x_bounds(m, rect):
        return m.x_min[rect], m.x_max[rect]

    def y_bounds(m, rect):
        return m.y_min[rect], m.y_max[rect]

    m.rect_x = Var(
        m.rectangles, doc="x-coordinate of rectangle center", bounds=x_bounds
    )
    m.rect_y = Var(
        m.rectangles, doc="y-coordinate of rectangle center", bounds=y_bounds
    )
    m.dist_x = Var(m.rect_pairs, doc="x-axis separation between rectangle pair")
    m.dist_y = Var(m.rect_pairs, doc="y-axis separation between rectangle pair")

    if metric == "l2":
        m.min_dist_cost = Objective(
            expr=sum(
                m.rect_sep_penalty[r1, r2]
                * (m.dist_x[r1, r2] ** 2 + m.dist_y[r1, r2] ** 2) ** 0.5
                for (r1, r2) in m.rect_pairs
            )
        )
    # l1 distance used in the paper
    else:
        m.min_dist_cost = Objective(
            expr=sum(
                m.rect_sep_penalty[r1, r2] * (m.dist_x[r1, r2] + m.dist_y[r1, r2])
                for (r1, r2) in m.rect_pairs
            )
        )

    # Ensure the dist_x and dist_y are greater than the positive and negative
    # signed distances.
    @m.Constraint(m.rect_pairs, doc="x-distance between rectangles")
    def dist_x_defn_1(m, r1, r2):
        return m.dist_x[r1, r2] >= m.rect_x[r2] - m.rect_x[r1]

    @m.Constraint(m.rect_pairs, doc="x-distance between rectangles")
    def dist_x_defn_2(m, r1, r2):
        return m.dist_x[r1, r2] >= m.rect_x[r1] - m.rect_x[r2]

    @m.Constraint(m.rect_pairs, doc="y-distance between rectangles")
    def dist_y_defn_1(m, r1, r2):
        return m.dist_y[r1, r2] >= m.rect_y[r2] - m.rect_y[r1]

    @m.Constraint(m.rect_pairs, doc="y-distance between rectangles")
    def dist_y_defn_2(m, r1, r2):
        return m.dist_y[r1, r2] >= m.rect_y[r1] - m.rect_y[r2]

    @m.Disjunction(
        m.rect_pairs,
        doc="Make sure that none of the rectangles overlap in "
        "either the x or y dimensions.",
    )
    def no_overlap(m, r1, r2):
        return [
            m.rect_x[r1] + m.rect_length[r1] / 2
            <= (m.rect_x[r2] - m.rect_length[r2] / 2),
            m.rect_y[r1] + m.rect_height[r1] / 2
            <= (m.rect_y[r2] - m.rect_height[r2] / 2),
            m.rect_x[r2] + m.rect_length[r2] / 2
            <= (m.rect_x[r1] - m.rect_length[r1] / 2),
            m.rect_y[r2] + m.rect_height[r2] / 2
            <= (m.rect_y[r1] - m.rect_height[r1] / 2),
        ]

    @m.Disjunction(m.rectangles, doc="Each rectangle must be in a circle.")
    def rectangle_in_circle(m, r):
        return [
            [
                # Rectangle lower left corner in circle
                (m.rect_x[r] - m.rect_length[r] / 2 - m.circle_x[c]) ** 2
                + (m.rect_y[r] + m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
                # rectangle upper left corner in circle
                (m.rect_x[r] - m.rect_length[r] / 2 - m.circle_x[c]) ** 2
                + (m.rect_y[r] - m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
                # rectangle lower right corner in circle
                (m.rect_x[r] + m.rect_length[r] / 2 - m.circle_x[c]) ** 2
                + (m.rect_y[r] + m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
                # rectangle upper right corner in circle
                (m.rect_x[r] + m.rect_length[r] / 2 - m.circle_x[c]) ** 2
                + (m.rect_y[r] - m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
            ]
            for c in m.circles
        ]

    return m


def draw_model(m, title=None):
    """Draw a model using matplotlib to illustrate what's going on. Pass 'title' argument to give chart a title"""

    # matplotlib setup
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Hardcode these bounds since I'm not sure the best way to do it automatically
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    if title is not None:
        plt.title(title)

    for c in m.circles:
        print(
            f"drawing circle {c}: x={m.circle_x[c]}, y={m.circle_y[c]}, r={m.circle_r[c]}"
        )
        # no idea about colors
        ax.add_patch(
            mpl.patches.Circle(
                (m.circle_x[c], m.circle_y[c]),
                radius=m.circle_r[c],
                facecolor="#0d98e6",
                edgecolor="#000000",
            )
        )
        ax.text(
            m.circle_x[c],
            m.circle_y[c] + m.circle_r[c] + 1.5,
            f"Circle {c}",
            horizontalalignment="center",
        )

    for r in m.rectangles:
        print(
            f"drawing rectangle {r}: x={value(m.rect_x[r])}, y={value(m.rect_y[r])} (center), L={m.rect_length[r]}, H={m.rect_height[r]}"
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (
                    value(m.rect_x[r]) - m.rect_length[r] / 2,
                    value(m.rect_y[r]) - m.rect_height[r] / 2,
                ),
                m.rect_length[r],
                m.rect_height[r],
                facecolor="#fbec68",
                edgecolor="#000000",
            )
        )
        ax.text(
            value(m.rect_x[r]),
            value(m.rect_y[r]),
            f"R{r}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )

    plt.show()


# Solve some example problems and draw some pictures
if __name__ == "__main__":
    from pyomo.environ import SolverFactory, TransformationFactory

    # Set up a solver, for example scip and bigm works
    solver = SolverFactory("scip")
    transformer = TransformationFactory("gdp.bigm")

    # Do all of the possible problems
    for d in ["l1", "l2"]:
        for key in constrained_layout_model_examples.keys():
            print(f"Solving example problem: {key}")
            model = build_constrained_layout_model(
                constrained_layout_model_examples[key], metric=d
            )
            transformer.apply_to(model)
            solver.solve(model)
            print(f"Found objective function value: {model.min_dist_cost()}")
            draw_model(model, title=(f"{key} ({d} distance)"))
            print()
