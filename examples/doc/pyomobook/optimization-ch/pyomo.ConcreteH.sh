#!/bin/sh

pyomo solve --solver=ipopt ConcreteH.py 
cat results.yml
rm -f results.yml results.json
