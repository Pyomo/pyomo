from abc import ABC, abstractmethod
from pyomo.core.base.block import _BlockData
from pyomo.contrib import appsi
from typing import Optional


class CutGenerator(ABC):
    @abstractmethod
    def generate(self, model: _BlockData, solver: Optional[appsi.base.Solver] = None):
        pass
