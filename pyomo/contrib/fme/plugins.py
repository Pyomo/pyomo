from pyomo.core.base import TransformationFactory
from .fourier_motzkin_elimination import \
    Fourier_Motzkin_Elimination_Transformation

def load():
    TransformationFactory.register(
        'contrib.fourier_motzkin_elimination', 
        doc="Project out specified (continuous) "
        "variables from a linear model.")(
            Fourier_Motzkin_Elimination_Transformation)
