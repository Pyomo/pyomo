"""Strip packing example from MINLP.org library.
Strip-packing example from http://minlp.org/library/lib.php?lib=GDP
This model packs a set of rectangles without rotation or overlap within a
strip of a given width, minimizing the length of the strip.
Common applications include stamping of components from a metal sheet or
cutting fabric.

Example taken from Systematic Modeling of Discrete-Continuous Optimization
Models through Generalized Disjunctive Programming paper, table 8.
DOI: 10.1002/aic.14088

"""

from __future__ import division

from pyomo.environ import (ConcreteModel, NonNegativeReals, Objective, Param,
                           Set, SolverFactory, TransformationFactory, Var)


# x and y are flipped from article
# not same rectangles as article
def build_rect_strip_packing_model():
    """Build the strip packing model."""
    model = ConcreteModel(name="Rectangles strip packing")
    model.rectangles = Set(ordered=True, initialize=[0, 1, 2, 3, 4, 5, 6, 7])

    # Width and Length of each rectangle
    model.rect_width = Param(
        model.rectangles, initialize={0: 3, 1: 3, 2: 2, 3: 2, 4: 3, 5: 5,
                                      6: 7, 7: 7})
    # parameter indexed by each rectangle
    # same as height?
    model.rect_length = Param(
        model.rectangles, initialize={0: 4, 1: 3, 2: 2, 3: 2, 4: 3, 5: 3,
                                      6: 4, 7: 4})

    model.strip_width = Param(
        initialize=10, doc="Available width of the strip")

    # upperbound on length (default is sum of lengths of rectangles)
    model.max_length = Param(
        initialize=sum(model.rect_length[i] for i in model.rectangles),
        doc="maximum length of the strip (if all rectangles were arranged "
        "lengthwise)")

    # x (length) and y (width) coordinates of each of the rectangles
    model.x = Var(model.rectangles,
                  bounds=(0, model.max_length),
                  doc="rectangle corner x-position (position across length)")

    def w_bounds(m, i):
        return (0, m.strip_width - m.rect_width[i])
    model.y = Var(model.rectangles,
                  bounds=w_bounds,
                  doc="rectangle corner y-position (position down width)")

    model.strip_length = Var(
        within=NonNegativeReals, doc="Length of strip required.")

    def rec_pairs_filter(model, i, j):
        return i < j
    model.overlap_pairs = Set(
        initialize=model.rectangles * model.rectangles,
        dimen=2, filter=rec_pairs_filter,
        doc="set of possible rectangle conflicts")

    @model.Constraint(model.rectangles)
    def strip_ends_after_last_rec(model, i):
        return model.strip_length >= model.x[i] + model.rect_length[i]

    model.total_length = Objective(expr=model.strip_length,
                                   doc="Minimize length")

    @model.Disjunction(
        model.overlap_pairs,
        doc="Make sure that none of the rectangles on the strip overlap in "
        "either the x or y dimensions.")
    def no_overlap(m, i, j):
        return [
            m.x[i] + m.rect_length[i] <= m.x[j],
            m.x[j] + m.rect_length[j] <= m.x[i],
            m.y[i] + m.rect_width[i] <= m.y[j],
            m.y[j] + m.rect_width[j] <= m.y[i],
        ]

    return model


if __name__ == "__main__":
    model = build_rect_strip_packing_model()
    TransformationFactory('gdp.chull').apply_to(model)
    opt = SolverFactory('gurobi')
    results = opt.solve(model, tee=True)
