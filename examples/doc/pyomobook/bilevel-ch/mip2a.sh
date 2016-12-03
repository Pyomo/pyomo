#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_blp_global \
            --solver-options="bigM=100 solver=glpk" \
            bard511.py
# @:pyomo
cat results.yml
rm results.yml
