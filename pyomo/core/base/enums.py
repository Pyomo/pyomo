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

import enum
import sys
from pyomo.common import enums

if sys.version_info[:2] >= (3, 11):
    strictEnum = {'boundary': enum.STRICT}
else:
    strictEnum = {}


class TraversalStrategy(enum.Enum, **strictEnum):
    BreadthFirstSearch = 1
    PrefixDepthFirstSearch = 2
    PostfixDepthFirstSearch = 3
    # aliases
    BFS = BreadthFirstSearch
    ParentLastDepthFirstSearch = PostfixDepthFirstSearch
    PostfixDFS = PostfixDepthFirstSearch
    ParentFirstDepthFirstSearch = PrefixDepthFirstSearch
    PrefixDFS = PrefixDepthFirstSearch
    DepthFirstSearch = PrefixDepthFirstSearch
    DFS = DepthFirstSearch


class SortComponents(enum.Flag, **strictEnum):
    """
    This class is a convenient wrapper for specifying various sort
    ordering.  We pass these objects to the "sort" argument to various
    accessors / iterators to control how much work we perform sorting
    the resultant list.  The idea is that
    "sort=SortComponents.deterministic" is more descriptive than
    "sort=True".
    """

    UNSORTED = 0
    # Note: skip '1' so that we can map True to something other than 1
    ORDERED_INDICES = 2
    SORTED_INDICES = 4
    ALPHABETICAL = 8

    # aliases
    # TODO: deprecate some of these
    unsorted = UNSORTED
    indices = SORTED_INDICES
    declOrder = UNSORTED
    declarationOrder = declOrder
    alphaOrder = ALPHABETICAL
    alphabeticalOrder = alphaOrder
    alphabetical = alphaOrder
    # both alpha and decl orders are deterministic, so only must sort indices
    deterministic = ORDERED_INDICES
    sortBoth = indices | alphabeticalOrder  # Same as True
    alphabetizeComponentAndIndex = sortBoth

    @classmethod
    def _missing_(cls, value):
        if type(value) is bool:
            if value:
                return cls.SORTED_INDICES | cls.ALPHABETICAL
            else:
                return cls.UNSORTED
        elif value is None:
            return cls.UNSORTED
        return super()._missing_(value)

    @staticmethod
    def default():
        return SortComponents.UNSORTED

    @staticmethod
    def sorter(sort_by_names=False, sort_by_keys=False):
        sort = SortComponents.default()
        if sort_by_names:
            sort |= SortComponents.ALPHABETICAL
        if sort_by_keys:
            sort |= SortComponents.SORTED_INDICES
        return sort

    @staticmethod
    def sort_names(flag):
        return SortComponents.ALPHABETICAL in SortComponents(flag)

    @staticmethod
    def sort_indices(flag):
        return SortComponents.SORTED_INDICES in SortComponents(flag)


class VarCollector(enums.IntEnum):
    FromVarComponents = 1
    FromExpressions = 2
