import pyutilib.misc
from six import StringIO

output = StringIO()
pyutilib.misc.setup_redirect(output)

try:
    # Run the runner
    from run_path_constraint import results, m as model
finally:
    pyutilib.misc.reset_redirect()

# Report the result
for line in output.getvalue().splitlines():
    if line.startswith('EXIT'):
        print(line)

model.obj.display()
model.u.display()
model.x1.display()
model.x2.display()
model.x3.display()
