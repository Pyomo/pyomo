#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_blp_global \
            --solver-options="bigM=100 solver=glpk" \
            bard511_explicit.py
# @:pyomo
cat results.yml
