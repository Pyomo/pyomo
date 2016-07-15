#!/bin/sh

pyomo solve --solver=ipopt DiseaseEstimationCallback.py DiseaseEstimation.dat
