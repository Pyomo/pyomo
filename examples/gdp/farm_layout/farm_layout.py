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
Farm layout example from Sawaya (2006). The goal is to determine optimal placements and dimensions for farm
plots of specified areas to minimize the perimeter of a minimal enclosing fence. This is a GDP problem with 
some hyperbolic constraints to establish consistency of areas with length and width. The FLay05 and FLay06 
instances may take some time to solve; the others should be fast. Note that the Sawaya paper contains a 
little bit of nonclarity: it references "height" variables which do not exist - we use "length" for the x-axis 
and "width" on the y-axis, and it also is unclear on the way the coordinates define the rectangles; we have
decided that they are on the bottom-left and adapted the disjunction constraints to match.
"""

from pyomo.environ import (
    ConcreteModel,
    Objective,
    Param,
    RangeSet,
    Set,
    Var,
    Constraint,
    NonNegativeReals,
)

# Format: areas, length lower bounds, width lower bounds, length overall upper bound, width overall upper bound
# Examples are from Sawaya (2006), except the labelled alternate ones.
farm_layout_model_examples = {
    "FLay02": {
        'areas': {1: 40, 2: 50},
        'length_lbs': {1: 1, 2: 1},
        'width_lbs': {1: 1, 2: 1},
        'length_upper_overall': 30,
        'width_upper_overall': 30,
    },
    "FLay03": {
        'areas': {1: 40, 2: 50, 3: 60},
        'length_lbs': {1: 1, 2: 1, 3: 1},
        'width_lbs': {1: 1, 2: 1, 3: 1},
        'length_upper_overall': 30,
        'width_upper_overall': 30,
    },
    "FLay04": {
        'areas': {1: 40, 2: 50, 3: 60, 4: 35},
        'length_lbs': {1: 3, 2: 3, 3: 3, 4: 3},
        'width_lbs': {1: 3, 2: 3, 3: 3, 4: 3},
        'length_upper_overall': 100,
        'width_upper_overall': 100,
    },
    "FLay05": {
        'areas': {1: 40, 2: 50, 3: 60, 4: 35, 5: 75},
        'length_lbs': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        'width_lbs': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        'length_upper_overall': 30,
        'width_upper_overall': 30,
    },
    "FLay06": {
        'areas': {1: 40, 2: 50, 3: 60, 4: 35, 5: 75, 6: 20},
        'length_lbs': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
        'width_lbs': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
        'length_upper_overall': 30,
        'width_upper_overall': 30,
    },
    # An example where the dimension constraints are strict enough that it cannot fit a square
    "FLay03_alt_1": {
        'areas': {1: 40, 2: 50, 3: 60},
        'length_lbs': {1: 5, 2: 5, 3: 5},
        'width_lbs': {1: 5, 2: 5, 3: 7},
        'length_upper_overall': 30,
        'width_upper_overall': 30,
    },
    "FLay03_alt_2": {
        'areas': {1: 40, 2: 50, 3: 60},
        'length_lbs': {1: 1, 2: 1, 3: 1},
        'width_lbs': {1: 1, 2: 1, 3: 1},
        'length_upper_overall': 8,
        'width_upper_overall': 30,
    },
}


def build_model(params=farm_layout_model_examples["FLay03"]):
    """Build the model. Default example gives FLay03 which is fairly small"""

    # Ensure good data was passed
    assert (
        len(params) == 5
    ), "Params should be in the format: areas, length_lbs, width_lbs, length_upper_overall, width_upper_overall"

    areas = params['areas']
    length_lbs = params['length_lbs']
    width_lbs = params['width_lbs']
    length_upper_overall = params['length_upper_overall']
    width_upper_overall = params['width_upper_overall']

    for a in areas:
        assert a > 0, "Area must be positive"

    m = ConcreteModel(name="Farm Layout model")
    m.plots = RangeSet(len(areas), doc=f"{len(areas)} plots")
    m.plot_pairs = Set(initialize=m.plots * m.plots, filter=lambda _, p1, p2: p1 < p2)

    m.length_lbs = Param(
        m.plots,
        initialize=length_lbs,
        doc="Lower bounds for length of individual rectangles",
    )
    m.width_lbs = Param(
        m.plots,
        initialize=width_lbs,
        doc="Lower bounds for length of individual rectangles",
    )
    m.plot_area = Param(m.plots, initialize=areas, doc="Area of this plot")

    m.overall_length_ub = Param(
        initialize=length_upper_overall, doc="Overall upper bound for length"
    )
    m.overall_width_ub = Param(
        initialize=width_upper_overall, doc="Overall upper bound for width"
    )

    m.Length = Var(
        domain=NonNegativeReals, doc="Overall length of the farmland configuration"
    )
    m.Width = Var(
        domain=NonNegativeReals, doc="Overall width of the farmland configuration"
    )

    m.perim = Objective(
        expr=2 * (m.Length + m.Width), doc="Perimeter of the farmland configuration"
    )

    m.plot_x = Var(
        m.plots,
        bounds=(0, m.overall_length_ub),
        doc="x-coordinate of plot bottom-left corner",
    )
    m.plot_y = Var(
        m.plots,
        bounds=(0, m.overall_width_ub),
        doc="y-coordinate of plot bottom-left corner",
    )
    m.plot_length = Var(
        m.plots, bounds=(0, m.overall_length_ub), doc="Length of this plot"
    )
    m.plot_width = Var(
        m.plots, bounds=(0, m.overall_width_ub), doc="Width of this plot"
    )

    # Constraints

    # Ensure consistency of Length with lengths and x-values of plots
    @m.Constraint(m.plots, doc="Length is consistent with lengths of plots")
    def length_consistency(m, p):
        return m.Length >= m.plot_x[p] + m.plot_length[p]

    # As above
    @m.Constraint(m.plots, doc="Width is consistent with widths of plots")
    def width_consistency(m, p):
        return m.Width >= m.plot_y[p] + m.plot_width[p]

    @m.Constraint(m.plots, doc="Lengths and widths are consistent with area")
    def area_consistency(m, p):
        return m.plot_area[p] / m.plot_width[p] - m.plot_length[p] <= 0

    @m.Disjunction(
        m.plot_pairs,
        doc="Make sure that none of the plots overlap in "
        "either the x or y dimensions.",
    )
    def no_overlap(m, p1, p2):
        return [
            m.plot_x[p1] + m.plot_length[p1] <= m.plot_x[p2],
            m.plot_y[p1] + m.plot_width[p1] <= m.plot_y[p2],
            m.plot_x[p2] + m.plot_length[p2] <= m.plot_x[p1],
            m.plot_y[p2] + m.plot_width[p2] <= m.plot_y[p1],
        ]

    m.length_cons = Constraint(expr=m.Length <= m.overall_length_ub)
    m.width_cons = Constraint(expr=m.Width <= m.overall_width_ub)

    # This imposes a square-ness constraint. I'm not sure the justification for these
    # upper bounds in particular
    @m.Constraint(m.plots, doc="Length bounds")
    def length_bounds(m, p):
        return (m.length_lbs[p], m.plot_length[p], m.plot_area[p] / m.length_lbs[p])

    @m.Constraint(m.plots, doc="Width bounds")
    def width_bounds(m, p):
        return (m.width_lbs[p], m.plot_width[p], m.plot_area[p] / m.width_lbs[p])

    # ensure compatibility between coordinates, l/w lower bounds, and overall upper bounds.
    @m.Constraint(m.plots, doc="x-coordinate compatibility")
    def x_compat(m, p):
        return m.plot_x[p] <= m.overall_length_ub - m.length_lbs[p]

    @m.Constraint(m.plots, doc="y-coordinate compatibility")
    def y_compat(m, p):
        return m.plot_y[p] <= m.overall_width_ub - m.width_lbs[p]

    return m


def draw_model(m, title=None):
    """Draw a model using matplotlib to illustrate what's going on. Pass 'title' arg to give chart a title"""

    # matplotlib setup
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set bounds
    ax.set_xlim([m.overall_length_ub() * -0.2, m.overall_length_ub() * 1.2])
    ax.set_ylim([m.overall_width_ub() * -0.2, m.overall_width_ub() * 1.2])

    if title is not None:
        plt.title(title)

    # First, let's transparently draw the overall bounds
    ax.add_patch(
        mpl.patches.Rectangle(
            (0, 0),
            m.overall_length_ub(),
            m.overall_width_ub(),
            facecolor="#229922",
            edgecolor="#000000",
            alpha=0.2,
        )
    )

    # Now draw the plots
    for p in m.plots:
        print(
            f"drawing plot {p}: x={m.plot_x[p]()}, y={m.plot_y[p]()}, length={m.plot_length[p]()}, width={m.plot_width[p]()}"
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (m.plot_x[p](), m.plot_y[p]()),
                m.plot_length[p](),
                m.plot_width[p](),
                facecolor="#ebdc78",
                edgecolor="#000000",
            )
        )
        ax.text(
            m.plot_x[p]() + m.plot_length[p]() / 2,
            m.plot_y[p]() + m.plot_width[p]() / 2,
            f"Plot {p}",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.show()


if __name__ == "__main__":
    from pyomo.environ import SolverFactory
    from pyomo.core.base import TransformationFactory

    # Set up a solver, for example scip and hull works.
    # Note that these constraints are not linear or quadratic.
    solver = SolverFactory("scip")
    transformer = TransformationFactory("gdp.hull")

    # Try all the instances except FLay05 and FLay06, since they take a while.
    for key in farm_layout_model_examples.keys():
        if key not in ["FLay06", "FLay05"]:
            print(f"solving example problem: {key}")
            print("building model")
            model = build_model(farm_layout_model_examples[key])
            print("applying transformer")
            transformer.apply_to(model)
            print("solving model")
            solver.solve(model)
            print(f"Found objective function value: {model.perim()}")
            draw_model(model, title=key)
            print()
