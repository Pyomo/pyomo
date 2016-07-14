#!/bin/sh

# @cmd:
pyomo solve --solver=glpk wl-concrete.py
# @:cmd
cat results.yml
