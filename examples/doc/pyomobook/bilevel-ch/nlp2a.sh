#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_blp_local \
            --solver-options="mpec_bound=0.01 solver=ipopt" \
            bard511.py
# @:pyomo
cat results.yml
rm results.yml
