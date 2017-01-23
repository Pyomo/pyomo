#!/bin/sh

# @cmd:
pyomo solve abstract5.yaml
# @:cmd
cat results.yml
rm -f results.yml results.json
