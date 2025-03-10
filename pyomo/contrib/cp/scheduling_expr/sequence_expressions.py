#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.logical_expr import BooleanExpression


class NoOverlapExpression(BooleanExpression):
    """
    Expression representing that none of the IntervalVars in a SequenceVar overlap
    (if they are scheduled)

    args:
        args (tuple): Child node of type SequenceVar
    """

    def nargs(self):
        return 1

    def _to_string(self, values, verbose, smap):
        return "no_overlap(%s)" % values[0]


class FirstInSequenceExpression(BooleanExpression):
    """
    Expression representing that the specified IntervalVar is the first in the
    sequence specified by SequenceVar (if it is scheduled)

    args:
        args (tuple): Child nodes, the first of type IntervalVar, the second of type
                      SequenceVar
    """

    def nargs(self):
        return 2

    def _to_string(self, values, verbose, smap):
        return "first_in(%s, %s)" % (values[0], values[1])


class LastInSequenceExpression(BooleanExpression):
    """
    Expression representing that the specified IntervalVar is the last in the
    sequence specified by SequenceVar (if it is scheduled)

    args:
        args (tuple): Child nodes, the first of type IntervalVar, the second of type
                      SequenceVar
    """

    def nargs(self):
        return 2

    def _to_string(self, values, verbose, smap):
        return "last_in(%s, %s)" % (values[0], values[1])


class BeforeInSequenceExpression(BooleanExpression):
    """
    Expression representing that one IntervalVar occurs before another in the
    sequence specified by the given SequenceVar (if both are scheduled)

    args:
        args (tuple): Child nodes, the IntervalVar that must be before, the
                      IntervalVar that must be after, and the SequenceVar
    """

    def nargs(self):
        return 3

    def _to_string(self, values, verbose, smap):
        return "before_in(%s, %s, %s)" % (values[0], values[1], values[2])


class PredecessorToExpression(BooleanExpression):
    """
    Expression representing that one IntervalVar is a direct predecessor to another
    in the sequence specified by the given SequenceVar (if both are scheduled)

    args:
        args (tuple): Child nodes, the predecessor IntervalVar, the successor
                      IntervalVar, and the SequenceVar
    """

    def nargs(self):
        return 3

    def _to_string(self, values, verbose, smap):
        return "predecessor_to(%s, %s, %s)" % (values[0], values[1], values[2])


def no_overlap(sequence_var):
    """
    Creates a new NoOverlapExpression

    Requires that none of the scheduled intervals in the SequenceVar overlap each other

    args:
        sequence_var: A SequenceVar
    """
    return NoOverlapExpression((sequence_var,))


def first_in_sequence(interval_var, sequence_var):
    """
    Creates a new FirstInSequenceExpression

    Requires that 'interval_var' be the first in the sequence specified by
    'sequence_var' if it is scheduled

    args:
        interval_var (IntervalVar): The activity that should be scheduled first
            if it is scheduled at all
        sequence_var (SequenceVar): The sequence of activities
    """
    return FirstInSequenceExpression((interval_var, sequence_var))


def last_in_sequence(interval_var, sequence_var):
    """
    Creates a new LastInSequenceExpression

    Requires that 'interval_var' be the last in the sequence specified by
    'sequence_var' if it is scheduled

    args:
        interval_var (IntervalVar): The activity that should be scheduled last
            if it is scheduled at all
        sequence_var (SequenceVar): The sequence of activities
    """

    return LastInSequenceExpression((interval_var, sequence_var))


def before_in_sequence(before_var, after_var, sequence_var):
    """
    Creates a new BeforeInSequenceExpression

    Requires that 'before_var' be scheduled to start before 'after_var' in the
    sequence specified bv 'sequence_var', if both are scheduled

    args:
        before_var (IntervalVar): The activity that should be scheduled earlier in
            the sequence
        after_var (IntervalVar): The activity that should be scheduled later in the
            sequence
        sequence_var (SequenceVar): The sequence of activities
    """
    return BeforeInSequenceExpression((before_var, after_var, sequence_var))


def predecessor_to(before_var, after_var, sequence_var):
    """
    Creates a new PredecessorToExpression

    Requires that 'before_var' be a direct predecessor to 'after_var' in the
    sequence specified by 'sequence_var', if both are scheduled

    args:
        before_var (IntervalVar): The activity that should be scheduled as the
            predecessor
        after_var (IntervalVar): The activity that should be scheduled as the
            successor
        sequence_var (SequenceVar): The sequence of activities
    """
    return PredecessorToExpression((before_var, after_var, sequence_var))
