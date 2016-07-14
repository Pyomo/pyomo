#!/bin/sh
echo `which pyomo`
# @cmd:
pyomo solve --solver=glpk wl_abstract.py wl_data.dat
# @:cmd
cat results.yml
