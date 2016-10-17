#!/bin/sh

# @cmd:
pyomo solve --solver=asl \
            --solver-options="solver=ipopt" simple.py
# @:cmd
cat results.yml
