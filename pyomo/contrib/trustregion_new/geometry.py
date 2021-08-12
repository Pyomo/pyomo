#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
For a quadratic reduced model, this generates geometry,
i.e., a subset of random points within a trust region
"""

import logging
import numpy as np
from io import StringIO
from pyomo.contrib.trustregion_new.cache import GeometryCache

logger = logging.getLogger('pyomo.contrib.trustregion_new')

def _points_to_matrix(points, geometryParam):
    """
    Converts a set of points into matrix representation

    Parameters
    ----------
    points : ndarray
        Set of randomly generated normal points
    geometryParam : Int or list
        Single or range of natural number geometry parameters

    Returns
    -------
    Matrix representation of set of normal points

    """
    matrix = []
    for point in points:
        basisValue = [1]
        for i in range(0, geometryParam):
            basisValue.append(point[i])
        for i in range(0, geometryParam):
            for j in range(i, geometryParam):
                basisValue.append(point[i]*point[j])
        matrix.append(basisValue)
    return np.matrix(matrix)


def generate_geometry(geometryParam, iterations=None):
    """
    Generate optimal random points for a given geometry.

    Parameters
    ----------
    geometryParam : Int or list
        Single value or range of natural number geometry parameters
    iterations : Int
        Number of iterations.

    Returns
    -------
    Optimal values for: condition number, set of points, and matrix

    """
    if geometryParam in GeometryCache:
        optimalConditionNumber, txt = GeometryCache[geometryParam]
        optimalPointSet = np.loadtxt(StringIO(txt))
        if optimalPointSet.ndim < 2:
            optimalPointSet = optimalPointSet.reshape(optimalPointSet.size, 1)
        optimalMatrix = _points_to_matrix(optimalPointSet, geometryParam)
        if iterations is None:
            logger.info('Loading cached geometry with condition number %f'
                        % (optimalConditionNumber,))
            return optimalConditionNumber, optimalPointSet, optimalMatrix
    else:
        optimalConditionNumber = np.inf
        optimalPointSet = None
        optimalMatrix = None

    if iterations is None:
        # Given that we rarely run this code, setting iterations to a 
        # reasonably high number
        iterations = 5000

    logger.info('Generating %d random geometries for geometry parameter %f'
                % (iterations, geometryParam))
    dimension = int((geometryParam*geometryParam + geometryParam*3)/2 + 1)
    mean = np.zeros(geometryParam)
    for i in range(0, iterations):
        points = np.random.multivariate_normal(mean,
                                               np.eye(geometryParam),
                                               dimension-1)
        for j in range(dimension-1):
            points[j] = points[j]/np.linalg.norm(points[j])
        points = np.append(points, [mean], axis=0)
        matrix = _points_to_matrix(points, geometryParam)
        conditionNumber = np.linalg.cond(matrix)
        if conditionNumber < optimalConditionNumber:
            logger.info("New condition number: %6d : %10.4f : %10.4f"
                        % (i, optimalConditionNumber, conditionNumber))
            optimalConditionNumber = conditionNumber
            optimalPointSet = points
            optimalMatrix = matrix
    if optimalPointSet is None:
        logger.error("Geometry parameter %d failed in initialization "
                     "(no non-singular geometries found)\n" % geometryParam)
    return optimalConditionNumber, optimalPointSet, optimalMatrix


















