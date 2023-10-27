from enum import IntEnum, Enum


class EigenValueBounder(IntEnum):
    Gershgorin = 1
    GershgorinWithSimplification = 2
    LinearProgram = 3
    Global = 4


class Effort(IntEnum):
    none = 0
    very_low = 1
    low = 2
    medium = 3
    high = 4
    very_high = 5


class RelaxationSide(IntEnum):
    UNDER = 1
    OVER = 2
    BOTH = 3


class FunctionShape(IntEnum):
    LINEAR = 1
    CONVEX = 2
    CONCAVE = 3
    UNKNOWN = 4
