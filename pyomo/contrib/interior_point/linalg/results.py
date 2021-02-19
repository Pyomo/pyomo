import enum


class LinearSolverStatus(enum.Enum):
    successful = 0
    not_enough_memory = 1
    singular = 2
    error = 3
    warning = 4


class LinearSolverResults(object):
    def __init__(self):
        self.status = None
