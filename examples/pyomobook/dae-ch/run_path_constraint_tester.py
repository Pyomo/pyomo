#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.tee import capture_output
from six import StringIO

output = StringIO()
capture = capture_output(output)
capture.setup()

try:
    # Run the runner
    from run_path_constraint import m as model
finally:
    capture.reset()

# Report the result
for line in output.getvalue().splitlines():
    if line.startswith('EXIT'):
        print(line)

model.obj.display()
model.u.display()
model.x1.display()
model.x2.display()
model.x3.display()
