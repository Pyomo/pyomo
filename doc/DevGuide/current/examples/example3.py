from pyomo.core import *

# @main:
@pyomo_api
def f3(data, x=None):
    """A simple example.

    Required:
        data.z: A nested required data value
        x: A required keyword argument
        x.y: A nested required data value

    Return:
        a: A return value
        b: Another return value
    """
    return PyomoAPIData(a=2*data.z, b=x.y)
# @:main

# @exec:
data = PyomoAPIData(z=1)
val = f3(data=data, x=PyomoAPIData(y=1))
# @:exec
print val
