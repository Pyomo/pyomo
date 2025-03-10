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


class PyNumeroEvaluationError(ArithmeticError):
    """An exception to be raised by PyNumero evaluation backends in the event
    of a failed function evaluation. This should be caught by solver interfaces
    and translated to the solver-specific evaluation error API.

    """

    pass
