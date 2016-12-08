#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_ld\
            --solver-options="bigM=100 solver=glpk"\
            interdiction.py
# @:pyomo
cat results.yml
