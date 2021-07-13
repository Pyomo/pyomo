#!/bin/sh

# @cmd:
pyomo solve --solver=glpk --solver-options='mipgap=0.01' concrete1.py
# @:cmd
cat results.yml
rm -f results.yml results.json
