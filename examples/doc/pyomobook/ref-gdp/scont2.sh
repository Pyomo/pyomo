#!/bin/sh

# @cmd:
pyomo solve scont2.py --transform gdp.bigm --solver=glpk --stream-solver
# @:cmd
cat results.yml
