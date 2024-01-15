#!/bin/sh

# @cmd:
pyomo convert --format=lp concrete1.py
# @:cmd
diff unknown.lp unknown-ref.lp
rm -f results.yml results.json unknown.lp
