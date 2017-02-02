#!/bin/sh

# @cmd:
pyomo solve --solver=ipopt --summary rosenbrock.py
# @:cmd
rm -f results.yml

