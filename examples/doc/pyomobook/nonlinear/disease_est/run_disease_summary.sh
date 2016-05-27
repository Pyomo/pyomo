#!/bin/sh

pyomo solve --solver=ipopt --summary DiseaseEstimation.py DiseaseEstimation.dat
