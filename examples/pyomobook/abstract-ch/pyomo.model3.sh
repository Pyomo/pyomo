#!/bin/sh

# @cmd:
pyomo convert --output=concrete1.nl concrete1.py
# @:cmd
python -m pyomo.repn.tests.nl_diff concrete1.nl concrete1-ref.nl
rm -f results.yml results.json concrete1.nl
