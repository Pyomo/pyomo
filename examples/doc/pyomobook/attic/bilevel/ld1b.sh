#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_ld\
            --solver-options="bigM=100 solver=glpk" interdiction_explicit.py
# @:pyomo
cat results.yml
