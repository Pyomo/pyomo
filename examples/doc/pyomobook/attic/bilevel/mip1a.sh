#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_blp_global \
            --solver-options="solver=glpk" bard511.py
# @:pyomo
cat results.yml
