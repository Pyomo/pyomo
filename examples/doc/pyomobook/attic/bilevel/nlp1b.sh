#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_blp_local \
            --solver-options="solver=ipopt" bard511_explicit.py
# @:pyomo
cat results.yml
