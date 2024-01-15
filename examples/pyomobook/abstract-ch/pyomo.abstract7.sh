#!/bin/sh

# @cmd:
pyomo solve --solver=glpk -c abstract7.py
# @:cmd
rm -f results.yml results.json abstract7.pyomo abstract7.results
