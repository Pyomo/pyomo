#!/bin/sh

# @cmd:
pyomo solve --solver=glpk --solver-suffix='.*' concrete2.py
# @:cmd
cat results.yml
\rm -f results.csv results.yml results.json
