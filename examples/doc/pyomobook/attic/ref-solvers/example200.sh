#!/bin/sh

# @cmd:
pyomo solve --solver=gurobi --solver-io=python simple.py
# @:cmd
cat results.yml
