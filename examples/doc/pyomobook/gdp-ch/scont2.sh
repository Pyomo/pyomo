#!/bin/sh

# @cmd:
pyomo solve scont2.py --transform gdp.bigm --solver=glpk
# @:cmd
cat results.yml
