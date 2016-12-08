#!/bin/sh

# @pyomo:
pyomo solve --solver=bilevel_blp_local \
            --solver-options="solver=ipopt" bard511.py
# @:pyomo
cat results.yml
rm results.yml
