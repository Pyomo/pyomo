#!/bin/sh

pyomo solve --solver=ipopt AbstractH.py AbstractH.dat
cat results.yml
