#!/bin/sh

# @cmd:
pyomo solve --solver=glpk wl_concrete.py
# @:cmd
cat results.yml
