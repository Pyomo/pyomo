#!/bin/sh
# @cmd:
pyomo solve --solver=ipopt --summary multimodal_init1.py
# @:cmd
rm -f results.yml
