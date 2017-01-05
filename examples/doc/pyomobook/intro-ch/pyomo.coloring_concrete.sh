#!/bin/sh

pyomo solve --solver=glpk coloring_concrete.py
rm -f results.yml results.json


