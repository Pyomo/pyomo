#!/bin/sh

# @cmd:
pyomo solve scont.py --transform gdp.bigm --solver=glpk
# @:cmd
cat results.yml
rm results.yml
