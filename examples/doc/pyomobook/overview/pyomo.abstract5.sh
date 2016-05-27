#!/bin/sh

# @cmd:
pyomo solve --solver=glpk abstract5.py abstract5.dat
# @:cmd
cat results.yml
