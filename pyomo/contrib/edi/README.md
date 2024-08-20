# Engineering Design Interface

The Pyomo Engineering Design Interface (EDI) is a lightweight wrapper on the Pyomo language that is targeted at composing engineering design optimization problems.  The language and interface have been designed to mimic many of the features found in [GPkit](https://github.com/convexengineering/gpkit) and [CVXPY](https://github.com/cvxpy/cvxpy) while also providing a simple, clean interface for black-box analysis codes that are common in engineering design applications.

## TODO
- Add unit tests for the BlackBox_Standardized method
- Add unit tests for the MultiCase method
- Add documentation for the MultiCase method
- Address Bethany's comments

## Installation

EDI is a part of the standard installation process for Pyomo:
```
pip install pyomo
```

EDI also requires the pint dependency that is optional in base Pyomo:
```
pip install pint
```

## Usage

The core object in EDI is the `Formulation`  object, which inherits from the `pyomo.environ.ConcreteModel`.  Essentially, a `Formulation` is a Pyomo `Model` with some extra stuff, but can be treated exactly as if it were a Pyomo `Model`.  However, an EDI `Formulation` has some additional features that can help simplify model construction.

<!-- Below is a simple example to get started, but additional resources can be found in the [examples](https://github.com/Pyomo/pyomo/tree/main/pyomo/contrib/edi/examples) folder or in the EDI [documentation](https://pyomo.readthedocs.io/en/stable/contributed_packages/edi/index.html) -->

```python
# =================
# Import Statements
# =================
import pyomo.environ as pyo
from pyomo.environ import units
from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'The unit circle output')

# =================
# Declare Constants
# =================
c = f.Constant(name = 'c', value = 1.0, units = '', description = 'A constant c', size = 2)

# =====================
# Declare the Objective
# =====================
f.Objective(
    c[0]*x + c[1]*y
)

# ===================
# Declare a Black Box
# ===================
class UnitCircle(BlackBoxFunctionModel):
    def __init__(self): # The initialization function
        
        # Initialize the black box model
        super().__init__()

        # A brief description of the model
        self.description = 'This model evaluates the function: z = x**2 + y**2'
        
        # Declare the black box model inputs
        self.inputs.append(name = 'x', units = 'ft' , description = 'The x variable')
        self.inputs.append(name = 'y', units = 'ft' , description = 'The y variable')

        # Declare the black box model outputs
        self.outputs.append(name = 'z', units = 'ft**2',  description = 'Resultant of the unit circle evaluation')

        # Declare the maximum available derivative
        self.availableDerivative = 1

        # Post-initialization setup
        self.post_init_setup()

    def BlackBox(self, x, y): # The actual function that does things
        x = pyo.value(units.convert(x,self.inputs['x'].units)) # Converts to correct units then casts to float
        y = pyo.value(units.convert(y,self.inputs['y'].units)) # Converts to correct units then casts to float

        z = x**2 + y**2 # Compute z
        dzdx = 2*x      # Compute dz/dx
        dzdy = 2*y      # Compute dz/dy

        z *= units.ft**2
        dzdx *= units.ft # units.ft**2 / units.ft
        dzdy *= units.ft # units.ft**2 / units.ft
        
        return z, [dzdx, dzdy] # return z, grad(z), hess(z)...

# =======================
# Declare the Constraints
# =======================
f.ConstraintList(
    [
        [ z, '==', [x,y], UnitCircle() ] ,
        z <= 1*units.m**2
    ]
)
```

## Acknowledgement

Pyomo: Python Optimization Modeling Objects  
Copyright (c) 2008-2023  
National Technology and Engineering Solutions of Sandia, LLC  
Under the terms of Contract DE-NA0003525 with National Technology and
Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
rights in this software.

Development of this module was conducted as part of the Institute for
the Design of Advanced Energy Systems (IDAES) with support through the
Simulation-Based Engineering, Crosscutting Research Program within the
U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.

This software is distributed under the 3-clause BSD License.