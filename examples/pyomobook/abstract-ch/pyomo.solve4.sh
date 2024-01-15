#!/bin/sh

# @cmd:
pyomo solve concrete1.yaml
# @:cmd
cat results.yml
rm -f results.yml results.json
