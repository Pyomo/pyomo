from pyomo.environ import *

# create a standard python dict for mapping subsquares to
# the list (row,col) entries
subsq_to_row_col = dict()

subsq_to_row_col[1] = [(i,j) for i in range(1,4) for j in range(1,4)]
subsq_to_row_col[2] = [(i,j) for i in range(1,4) for j in range(4,7)]
subsq_to_row_col[3] = [(i,j) for i in range(1,4) for j in range(7,10)]

subsq_to_row_col[4] = [(i,j) for i in range(4,7) for j in range(1,4)]
subsq_to_row_col[5] = [(i,j) for i in range(4,7) for j in range(4,7)]
subsq_to_row_col[6] = [(i,j) for i in range(4,7) for j in range(7,10)]

subsq_to_row_col[7] = [(i,j) for i in range(7,10) for j in range(1,4)]
subsq_to_row_col[8] = [(i,j) for i in range(7,10) for j in range(4,7)]
subsq_to_row_col[9] = [(i,j) for i in range(7,10) for j in range(7,10)]

# creates the sudoku model for a 10x10 board, where the
# input board is a list of fixed numbers specified in
# (row, col, val) tuples.
def create_sudoku_model(board):

    model = ConcreteModel()

    # store the starting board for the model
    model.board = board

    # create sets for rows columns and squares
    model.ROWS = RangeSet(1,9)
    model.COLS = RangeSet(1,9)
    model.SUBSQUARES = RangeSet(1,9)
    model.VALUES = RangeSet(1,9)

    # create the binary variables to define the values
    model.y = Var(model.ROWS, model.COLS, model.VALUES, within=Binary)

    # fix variables based on the current board
    for (r,c,v) in board:
        model.y[r,c,v].fix(1)

    # create the objective - this is a feasibility problem
    # so we just make it a constant
    model.obj = Objective(expr= 1.0)

# @row_col_cons:
    # exactly one number in each row
    def _RowCon(model, r, v):
        return sum(model.y[r,c,v] for c in model.COLS) == 1
    model.RowCon = Constraint(model.ROWS, model.VALUES, rule=_RowCon)

    # exactly one nubmer in each column
    def _ColCon(model, c, v):
        return sum(model.y[r,c,v] for r in model.ROWS) == 1
    model.ColCon = Constraint(model.COLS, model.VALUES, rule=_ColCon)
# @:row_col_cons

# @subsq_con:
    # exactly one number in each subsquare
    def _SubSqCon(model, s, v):
        return sum(model.y[r,c,v] for (r,c) in subsq_to_row_col[s]) == 1
    model.SubSqCon = Constraint(model.SUBSQUARES, model.VALUES, rule=_SubSqCon)
# @:subsq_con

# @num_con:
    # exactly one number in each cell
    def _ValueCon(model, r, c):
        return sum(model.y[r,c,v] for v in model.VALUES) == 1
    model.ValueCon = Constraint(model.ROWS, model.COLS, rule=_ValueCon)
# @:num_con

    return model

# use this function to add a new integer cut to the model.
def add_integer_cut(model):
    if not hasattr(model, "IntegerCuts"):
        model.IntegerCuts = ConstraintList()

    cut_expr = 0.0
    for r in model.ROWS:
        for c in model.COLS:
            for v in model.VALUES:
                if not model.y[r,c,v].fixed:
                    # check if the binary variable is on or off
                    # note, it may not be exactly 1
                    if value(model.y[r,c,v]) >= 0.5:
                        cut_expr += (1.0 - model.y[r,c,v])
                    else:
                        cut_expr += model.y[r,c,v]
    model.IntegerCuts.add(cut_expr >= 1)

# prints the current solution stored in the model
def print_solution(model):
    for r in model.ROWS:
        print('   '.join(str(v) for c in model.COLS
                         for v in model.VALUES
                         if value(model.y[r,c,v]) >= 0.5))
