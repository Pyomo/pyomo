#!/bin/sh

# @cmd:
pyomo solve --solver=ipopt --summary DeerProblem.py DeerProblem.dat
# @:cmd
rm -f results.yml
