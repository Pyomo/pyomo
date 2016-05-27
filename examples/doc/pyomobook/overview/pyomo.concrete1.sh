#!/bin/sh

# @cmd:
pyomo solve --solver=glpk concrete1.py
# @:cmd
cat results.yml
