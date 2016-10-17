#!/bin/sh

# @cmd:
pyomo solve --solver=py:gurobi simple.py
# @:cmd
cat results.yml
