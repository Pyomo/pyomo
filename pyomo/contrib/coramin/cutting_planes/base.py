from abc import ABC, abstractmethod
from pyomo.core.base.block import _BlockData
from pyomo.contrib import appsi
from typing import Optional
import pybnb


class CutGenerator(ABC):
    @abstractmethod
    def generate(self, node: Optional[pybnb.Node]):
        pass
