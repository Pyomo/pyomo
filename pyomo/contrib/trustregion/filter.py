#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


class FilterElement:
    """
    Filter element for comparison of feasibility
    """

    def __init__(self, objective, infeasible):
        self.objective = objective
        self.infeasible = infeasible

    def compare(self, element):
        if (element.objective >= self.objective and
            element.infeasible >= self.infeasible):
            return -1
        if (element.objective <= self.objective and
            element.infeasible <= self.infeasible):
            return 1
        return 0


class Filter:
    """
    Trust region filter
    
    Based on original filter by Eason, Biegler (2016)
    """

    TrustRegionFilter = []

    def addToFilter(self, filterElement):
        filterCopy = list(self.TrustRegionFilter)
        for i in filterCopy:
            if (i.compare(filterElement) == 1):
                self.TrustRegionFilter.remove(i)
            elif (i.compare(filterElement) == -1):
                return
        self.TrustRegionFilter.append(filterElement)

    def isAcceptable(self, filterElement, theta_max):
        if (filterElement.infeasible > theta_max):
            return False
        for element in self.TrustRegionFilter:
            if (element.compare(filterElement) == -1):
                return False
        return True
