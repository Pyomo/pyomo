"""2-D constrained layout example.

Example based on: https://www.minlp.org/library/problem/index.php?i=107&lib=GDP
Description can be found in: https://doi.org/10.1016/j.ejor.2011.10.002

This model attempts to pack a set of rectangles within a set of circles while
minimizing the cost of connecting the rectangles, a function of the distance
between the rectangle centers. It is assumed that the circles do not overlap
with each other.

"""
from __future__ import division

from pyomo.environ import (ConcreteModel, Objective, Param,
                           RangeSet, Set, Var)


def build_constrained_layout_model():
    """Build the model."""
    m = ConcreteModel(name="2-D constrained layout")
    m.rectangles = RangeSet(3, doc="Three rectangles")
    m.circles = RangeSet(2, doc="Two circles")

    m.rect_length = Param(
        m.rectangles, initialize={1: 5, 2: 7, 3: 3},
        doc="Rectangle length")
    m.rect_height = Param(
        m.rectangles, initialize={1: 6, 2: 5, 3: 3},
        doc="Rectangle height")

    m.circle_x = Param(m.circles, initialize={1: 15, 2: 50},
                       doc="x-coordinate of circle center")
    m.circle_y = Param(m.circles, initialize={1: 10, 2: 80},
                       doc="y-coordinate of circle center")
    m.circle_r = Param(m.circles, initialize={1: 6, 2: 5},
                       doc="radius of circle")

    @m.Param(m.rectangles, doc="Minimum feasible x value for rectangle")
    def x_min(m, rect):
        return min(
            m.circle_x[circ] - m.circle_r[circ] + m.rect_length[rect] / 2
            for circ in m.circles)

    @m.Param(m.rectangles, doc="Maximum feasible x value for rectangle")
    def x_max(m, rect):
        return max(
            m.circle_x[circ] + m.circle_r[circ] - m.rect_length[rect] / 2
            for circ in m.circles)

    @m.Param(m.rectangles, doc="Minimum feasible y value for rectangle")
    def y_min(m, rect):
        return min(
            m.circle_y[circ] - m.circle_r[circ] + m.rect_height[rect] / 2
            for circ in m.circles)

    @m.Param(m.rectangles, doc="Maximum feasible y value for rectangle")
    def y_max(m, rect):
        return max(
            m.circle_y[circ] + m.circle_r[circ] - m.rect_height[rect] / 2
            for circ in m.circles)

    m.ordered_rect_pairs = Set(
        initialize=m.rectangles * m.rectangles,
        filter=lambda _, r1, r2: r1 != r2)
    m.rect_pairs = Set(initialize=[
        (r1, r2) for r1, r2 in m.ordered_rect_pairs
        if r1 < r2])

    m.rect_sep_penalty = Param(
        m.rect_pairs, initialize={(1, 2): 300, (1, 3): 240, (2, 3): 100},
        doc="Penalty for separation distance between two rectangles.")

    def x_bounds(m, rect):
        return m.x_min[rect], m.x_max[rect]

    def y_bounds(m, rect):
        return m.y_min[rect], m.y_max[rect]

    m.rect_x = Var(
        m.rectangles, doc="x-coordinate of rectangle center",
        bounds=x_bounds)
    m.rect_y = Var(
        m.rectangles, doc="y-coordinate of rectangle center",
        bounds=y_bounds)
    m.dist_x = Var(
        m.rect_pairs, doc="x-axis separation between rectangle pair")
    m.dist_y = Var(
        m.rect_pairs, doc="y-axis separation between rectangle pair")

    m.min_dist_cost = Objective(
        expr=sum(m.rect_sep_penalty[r1, r2] *
                 (m.dist_x[r1, r2] + m.dist_y[r1, r2])
                 for (r1, r2) in m.rect_pairs))

    @m.Constraint(m.ordered_rect_pairs,
                  doc="x-distance between rectangles")
    def dist_x_defn(m, r1, r2):
        return m.dist_x[
            tuple(sorted([r1, r2]))] >= m.rect_x[r2] - m.rect_x[r1]

    @m.Constraint(m.ordered_rect_pairs,
                  doc="y-distance between rectangles")
    def dist_y_defn(m, r1, r2):
        return m.dist_y[
            tuple(sorted([r1, r2]))] >= m.rect_y[r2] - m.rect_y[r1]

    @m.Disjunction(
        m.rect_pairs,
        doc="Make sure that none of the rectangles overlap in "
        "either the x or y dimensions.")
    def no_overlap(m, r1, r2):
        return [
            m.rect_x[r1] + m.rect_length[r1] / 2 <= (
                m.rect_x[r2] - m.rect_length[r2] / 2),
            m.rect_y[r1] + m.rect_height[r1] / 2 <= (
                m.rect_y[r2] - m.rect_height[r2] / 2),
            m.rect_x[r2] + m.rect_length[r2] / 2 <= (
                m.rect_x[r1] - m.rect_length[r1] / 2),
            m.rect_y[r2] + m.rect_height[r2] / 2 <= (
                m.rect_y[r1] - m.rect_height[r1] / 2),
        ]

    @m.Disjunction(m.rectangles, doc="Each rectangle must be in a circle.")
    def rectangle_in_circle(m, r):
        return [
            [
                # Rectangle lower left corner in circle
                (m.rect_x[r] - m.rect_length[r] / 2 - m.circle_x[c]) ** 2 +
                (m.rect_y[r] + m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
                # rectangle upper left corner in circle
                (m.rect_x[r] - m.rect_length[r] / 2 - m.circle_x[c]) ** 2 +
                (m.rect_y[r] - m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
                # rectangle lower right corner in circle
                (m.rect_x[r] + m.rect_length[r] / 2 - m.circle_x[c]) ** 2 +
                (m.rect_y[r] + m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
                # rectangle upper right corner in circle
                (m.rect_x[r] + m.rect_length[r] / 2 - m.circle_x[c]) ** 2 +
                (m.rect_y[r] - m.rect_height[r] / 2 - m.circle_y[c]) ** 2
                <= m.circle_r[c] ** 2,
            ]
            for c in m.circles
        ]

    return m
