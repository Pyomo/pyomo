from pyomo.core.base.numvalue import NumericValue

class MockFixedValue(NumericValue):
    value = 42
    def __init__(self, v = 42):
        self.value = v
    def is_fixed(self):
        return True
    def __call__(self, exception=True):
        return self.value
