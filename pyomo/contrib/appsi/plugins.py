from pyomo.common.extensions import ExtensionBuilderFactory
from .build import AppsiBuilder

def load():
    ExtensionBuilderFactory.register('appsi')(AppsiBuilder)
