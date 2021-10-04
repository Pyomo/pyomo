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

logger = logging.getLogger('pyomo.contrib.trustregion_new')


def quadraticExpression(self, len_x):
    """
    Expression for use in quadratic RM type
    """
    return int((len_x*len_x + len_x*3)/2. + 1)


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


def generate_geometry(geometryParam, iterations):
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
        logger.info('Loading cached geometry with condition number %f'
                        % (optimalConditionNumber,))
        optimalPointSet = np.loadtxt(StringIO(txt))
        if optimalPointSet.ndim < 2:
            optimalPointSet = optimalPointSet.reshape(optimalPointSet.size, 1)
        optimalMatrix = _points_to_matrix(optimalPointSet, geometryParam)
    else:
        optimalConditionNumber = np.inf
        optimalPointSet = None
        optimalMatrix = None

    logger.info('Generating %d random geometries for geometry parameter %f'
                % (iterations, geometryParam))
    dimension = quadraticExpression(geometryParam)
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


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    import sys
    import os
    from pyomo.common.fileutils import this_file_dir
    try:
        from pyomo.contrib.trustregion_new.cache import GeometryCache
    except ImportError:
        GeometryCache = {}
    if len(sys.argv) < 2:
        print("""Usage: %s [GEOMETRYPARAM] [ITERATIONS]\n
                    GEOMETRYPARAM : single number or range in the form NN:NN (non-zero)\n
                    ITERATIONS    : number of random matrices to generate""")
    cache_file = os.path.join(this_file_dir(), 'cache.py')
    if not os.path.isfile(cache_file) or not os.access(cache_file, os.W_OK):
        logger.error(
            "Cannot write to the geometry cache (%s).  "
            "This utility is only expected to be run as a script in "
            "editable (development) source trees." % (cache_file,))
        sys.exit()

    if len(sys.argv) > 1:
        if ':' in sys.argv[1]:
            geomParamSet = range(*tuple(int(i) for i in sys.argv[1].split(':')))
        else:
            geomParamSet = [int(sys.argv[1])]
    else:
        geomParamSet = range(1, 24)
    if len(sys.argv) > 2:
        iterations = int(sys.argv[2])
    else:
        iterations = None
    for geometryParam in geomParamSet:
        local_iter = iterations if iterations else 10000*geometryParam
        conditionNumber, pointSet, matrix = generate_geometry(geometryParam, local_iter)
        if pointSet is not None:
            txt = StringIO()
            np.savetxt(txt, pointSet)
            GeometryCache[geometryParam] = (conditionNumber, txt.getvalue())
            with open(cache_file, 'w') as file:
                file.write("""
# ___________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ___________________________________________________________________________
#
# Cache of autogenerated quadratic ROM geometries
#
# THIS FILE IS AUTOGENERATED BY pyomo.contrib.trustregion.geometry
#
# DO NOT EDIT BY HAND
#
GeometryCache = {
""")
                for i in sorted(GeometryCache):
                    file.write('    %d: ( %f, """%s""" ),\n'
                               % (i, GeometryCache[i][0], GeometryCache[i][1]))
                file.write("}\n")
            logger.info("Condition number: geometryParam = %d is %f\n" % (geometryParam, conditionNumber))
