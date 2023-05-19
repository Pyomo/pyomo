"""
Farm layout example from Sawaya (2006)
"""

from __future__ import division

from pyomo.environ import ConcreteModel, Objective, Param, RangeSet, Set, Var, Constraint, NonNegativeReals

farm_layout_model_examples = {
    "Flay02": [
        {1: 40, 2: 50},
        {1: 1, 2: 1},
        {1: 1, 2: 1},
        30,
        30,
    ]
}

def build_model(areas, length_lbs, width_lbs, length_upper_overall, width_upper_overall):
    """Build the model."""
    
    # Ensure good data was passed
    for a in areas:
        assert a > 0, "Area must be positive"
    
    m = ConcreteModel(name="Farm Layout model")
    m.plots = RangeSet(len(areas), doc=f"{len(areas)} plots")
    m.plot_pairs = Set(
        initialize=m.plots * m.plots, filter=lambda _, p1, p2: p1 < p2
    )
    
    m.length_lbs = Param(m.plots, initialize=length_lbs, doc="Lower bounds for length of individual rectangles")
    m.width_lbs = Param(m.plots, initialize=width_lbs, doc="Lower bounds for length of individual rectangles")
    m.plot_area = Param(m.plots, initialize=areas, doc="Area of this plot")
    
    m.overall_length_ub = Param(initialize=length_upper_overall, doc="Overall upper bound for length")
    m.overall_width_ub = Param(initialize=width_upper_overall, doc="Overall upper bound for width")

    m.Length = Var(domain=NonNegativeReals, doc="Overall length we are minimizing")
    m.Width = Var(domain=NonNegativeReals, doc="Overall width we are minimizing")
    
    m.perim = Objective(expr=2 * (m.Length + m.Width), doc="Perimeter of the farmland configuration")

    m.plot_x = Var(m.plots, bounds=(0, m.overall_length_ub), doc="x-coordinate of plot center")
    m.plot_y = Var(m.plots, bounds=(0, m.overall_width_ub), doc="y-coordinate of plot center")
    m.plot_length = Var(m.plots, bounds=(0, m.overall_length_ub), doc="Length of this plot")
    m.plot_width = Var(m.plots, bounds=(0, m.overall_width_ub), doc="Width of this plot")

    # Constraints

    # I think this constraint is actually wrong in the paper. Shouldn't
    # it be plot_length / 2, since plot_x is the middle of the rectangle?
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
            m.plot_x[p1] + m.plot_length[p1] / 2
            <= (m.plot_x[p2] - m.plot_length[p2] / 2),
            m.plot_y[p1] + m.plot_width[p1] / 2
            <= (m.plot_y[p2] - m.plot_width[p2] / 2),
            m.plot_x[p2] + m.plot_length[p2] / 2
            <= (m.plot_x[p1] - m.plot_length[p1] / 2),
            m.plot_y[p2] + m.plot_width[p2] / 2
            <= (m.plot_y[p1] - m.plot_width[p1] / 2),
        ]
    m.length_cons = Constraint(expr=m.Length <= m.overall_length_ub)
    m.width_cons = Constraint(expr=m.Width <= m.overall_width_ub)

    # what do these do exactly? Shouldn't this be dividing by width_lbs ??
    @m.Constraint(m.plots, doc="Length bounds?")
    def length_bounds(m, p):
        return (m.length_lbs[p], m.plot_length[p], m.plot_area[p] / m.length_lbs[p])
    @m.Constraint(m.plots, doc="Width bounds?")
    def width_bounds(m, p):
        return (m.width_lbs[p], m.plot_width[p],  m.plot_area[p] / m.width_lbs[p])

    # ensure compatibility between coordinates, l/w lower bounds, and overall upper bounds.
    # But I think this is also wrong. Shouldn't it use L_i^L / 2 ?
    @m.Constraint(m.plots, doc="x-coordinate compatibility")
    def x_compat(m, p):
        return m.plot_x[p] <= m.overall_length_ub - m.length_lbs[p]
    @m.Constraint(m.plots, doc="y-coordinate compatibility")
    def y_compat(m, p):
        return m.plot_y[p] <= m.overall_width_ub - m.width_lbs[p]

    return m

if __name__ == "__main__":
    from pyomo.environ import SolverFactory
    from pyomo.core.base import TransformationFactory

    # Set up a solver, for example scip and bigm works
    solver = SolverFactory("scip")
    transformer = TransformationFactory("gdp.bigm")

    print("building model")
    model = build_model(*farm_layout_model_examples["Flay02"])
    print("applying transformer")
    transformer.apply_to(model)
    print("solving example problem")
    solver.solve(model)
    print(f"Found objective function value: {model.perim()}")
    print()
