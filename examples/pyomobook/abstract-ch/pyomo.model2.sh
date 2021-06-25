#!/bin/sh

# @cmd:
pyomo convert --output=concrete1.lp concrete1.py
# @:cmd
diff concrete1.lp ../command-ch/concrete1-ref.lp
rm -f results.yml results.json concrete1.lp
