#!/bin/sh

# @pyomo:
pyomo solve --solver=glpk buildactions.py buildactions_fails.dat
# @:pyomo
rm -f results.json results.yml
