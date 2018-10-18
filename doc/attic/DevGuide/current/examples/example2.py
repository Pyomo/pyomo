from pyomo.core import *

# @main:
@pyomo_api
def f2(x=0, y=1, data=None):
    """A simple example.

    Required:
        data: The required data argument
        x: A required keyword argument

    Optional:
        y: An optional keyword argument

    Return:
        a: A return value
        b: Another return value
    """
    return PyomoAPIData(a=2*data.z, b=x+y)
# @:main

# @exec:
data = PyomoAPIData(z=1)
val = f2(data=data, x=2)
# @:exec
print val
