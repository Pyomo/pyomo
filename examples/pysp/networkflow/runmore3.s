if [ "$1" == "" ]; then
   solver='gurobi'
else
   solver=$1
fi

echo "running lagrangeMorePR.py for 3-scenario network flow at $(date)$ using solver $solver" > runmore3.out
python ../lagrangeMorePR.py -m CCmodels -i CCdata/1ef3CC --solver=$solver --csvPrefix='run3' --prob-file='plist.dat' >> runmore3.out
vi runmore3.out
