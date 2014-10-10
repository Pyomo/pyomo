f [ "$1" == "" ]; then
   solver='gurobi'
else
   solver=$1
fi

echo "running lagrangeMorePR.py for 10-scenario network flow at $(date)$ using solver $solver" > runmore10.out
python ../lagrangeMorePR.py -m CCmodels -i CCdata/notuniform10 --solver=$solver --csvPrefix='10' >> runmore10.out
vi runmore10.out
