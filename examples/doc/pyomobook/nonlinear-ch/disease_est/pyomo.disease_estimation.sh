#!/bin/sh

# @cmd:
pyomo solve --solver=ipopt --logging=quiet disease_estimation.py disease_estimation.dat
# @:cmd
rm -f results.yml
