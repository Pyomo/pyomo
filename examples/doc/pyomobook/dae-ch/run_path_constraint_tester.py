from pyomo.common.tee import capture_output
from six import StringIO

output = StringIO()
capture = capture_output(output)
capture.setup()

try:
    # Run the runner
    from run_path_constraint import results, m as model
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
