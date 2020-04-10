#!/usr/bin/env bash

BASEDIR=`pwd`
WORKDIR=tmp_validate

function verify(){
    cd $WORKDIR
    cp ../$1 $2
    ERROR=
    patch < `echo ../$2 | sed 's/\.yml/.patch/'` || ERROR='patch failed'
    if test `diff ../$2 $2 | wc -l` -gt 0; then
        ERROR='diff inconsistent'
    fi
    if test -n "$ERROR"; then
        echo "$2 is in an inconsistent state: $ERROR"
    else
        rm $2
    fi
    cd $BASEDIR
}

mkdir $WORKDIR
verify unix_python_matrix_test.yml push_branch_unix_test.yml
verify unix_python_matrix_test.yml mpi_matrix_test.yml
verify win_python_matrix_test.yml push_branch_win_test.yml
rmdir --ignore-fail-on-non-empty $WORKDIR
