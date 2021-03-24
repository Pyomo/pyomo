#!/bin/sh

# @cmd:
pyomo solve --solver=glpk wl_abstract.py wl_data.dat
# @:cmd
cat results.yml
rm -f results.yml results.json
