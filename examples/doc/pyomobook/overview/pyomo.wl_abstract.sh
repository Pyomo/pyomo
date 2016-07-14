#!/bin/sh

# @cmd:
pyomo solve --solver=glpk wl-abstract.py wl-data.dat
# @:cmd
cat results.yml
