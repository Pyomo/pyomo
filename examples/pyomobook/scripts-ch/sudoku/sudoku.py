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

# create a standard python dict for mapping subsquares to
# the list (row,col) entries
subsq_to_row_col = dict()

subsq_to_row_col[1] = [(i, j) for i in range(1, 4) for j in range(1, 4)]
subsq_to_row_col[2] = [(i, j) for i in range(1, 4) for j in range(4, 7)]
subsq_to_row_col[3] = [(i, j) for i in range(1, 4) for j in range(7, 10)]

subsq_to_row_col[4] = [(i, j) for i in range(4, 7) for j in range(1, 4)]
subsq_to_row_col[5] = [(i, j) for i in range(4, 7) for j in range(4, 7)]
subsq_to_row_col[6] = [(i, j) for i in range(4, 7) for j in range(7, 10)]

subsq_to_row_col[7] = [(i, j) for i in range(7, 10) for j in range(1, 4)]
subsq_to_row_col[8] = [(i, j) for i in range(7, 10) for j in range(4, 7)]
subsq_to_row_col[9] = [(i, j) for i in range(7, 10) for j in range(7, 10)]


# creates the sudoku model for a 10x10 board, where the
# input board is a list of fixed numbers specified in
# (row, col, val) tuples.
def create_sudoku_model(board):
    model = pyo.ConcreteModel()

    # store the starting board for the model
    model.board = board

    # create sets for rows columns and squares
    model.ROWS = pyo.RangeSet(1, 9)
    model.COLS = pyo.RangeSet(1, 9)
    model.SUBSQUARES = pyo.RangeSet(1, 9)
    model.VALUES = pyo.RangeSet(1, 9)

    # create the binary variables to define the values
    model.y = pyo.Var(model.ROWS, model.COLS, model.VALUES, within=pyo.Binary)

    # fix variables based on the current board
    for r, c, v in board:
        model.y[r, c, v].fix(1)

    # create the objective - this is a feasibility problem
    # so we just make it a constant
    model.obj = pyo.Objective(expr=1.0)

    # @row_col_cons:
    # exactly one number in each row
    def _RowCon(model, r, v):
        return sum(model.y[r, c, v] for c in model.COLS) == 1

    model.RowCon = pyo.Constraint(model.ROWS, model.VALUES, rule=_RowCon)

    # exactly one number in each column
    def _ColCon(model, c, v):
        return sum(model.y[r, c, v] for r in model.ROWS) == 1

    model.ColCon = pyo.Constraint(model.COLS, model.VALUES, rule=_ColCon)
    # @:row_col_cons

    # @subsq_con:
    # exactly one number in each subsquare
    def _SqCon(model, s, v):
        return sum(model.y[r, c, v] for (r, c) in subsq_to_row_col[s]) == 1

    model.SqCon = pyo.Constraint(model.SUBSQUARES, model.VALUES, rule=_SqCon)
    # @:subsq_con

    # @num_con:
    # exactly one number in each cell
    def _ValueCon(model, r, c):
        return sum(model.y[r, c, v] for v in model.VALUES) == 1

    model.ValueCon = pyo.Constraint(model.ROWS, model.COLS, rule=_ValueCon)
    # @:num_con

    return model


# use this function to add a new integer cut to the model.
def add_integer_cut(model):
    # add the ConstraintList to store the IntegerCuts if
    # it does not already exist
    if not hasattr(model, "IntegerCuts"):
        model.IntegerCuts = pyo.ConstraintList()

    # add the integer cut corresponding to the current
    # solution in the model
    cut_expr = 0.0
    for r in model.ROWS:
        for c in model.COLS:
            for v in model.VALUES:
                if not model.y[r, c, v].fixed:
                    # check if the binary variable is on or off
                    # note, it may not be exactly 1
                    if pyo.value(model.y[r, c, v]) >= 0.5:
                        cut_expr += 1.0 - model.y[r, c, v]
                    else:
                        cut_expr += model.y[r, c, v]
    model.IntegerCuts.add(cut_expr >= 1)


# prints the current solution stored in the model
def print_solution(model):
    for r in model.ROWS:
        print(
            '   '.join(
                str(v)
                for c in model.COLS
                for v in model.VALUES
                if pyo.value(model.y[r, c, v]) >= 0.5
            )
        )
