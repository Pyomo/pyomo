#!/bin/sh

# @cmd:
pyomo solve --solver=ipopt --summary Rosenbrock.py
# @:cmd
rm -f results.yml

