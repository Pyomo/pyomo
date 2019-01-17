#!/usr/bin/env bash
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

if test "$1" = "--insecure"; then
    DOWNLOADERS=( "wget --no-check-certificate" "curl --insecure -L -O" )
else
    DOWNLOADERS=( "wget" "curl -L -O" )
fi
# Insecure wget: --no-check-certificate
# Insecure curl: --insecure

ASL=1.3.0
TARGET=https://github.com/ampl/mp/archive/$ASL.tar.gz

DOWNLOAD=
DOWNLOADERS=( "wget" "curl -L -O" )
for test_cmd in "${DOWNLOADERS[@]}"; do
    echo $test_cmd
    $test_cmd --help > /dev/null 2>&1
    if test $? -eq 0; then
        DOWNLOAD="$test_cmd"
        break
    fi
done
if test -z "$DOWNLOAD"; then
    echo "ERROR: no downloader found. Tried:"
    for test_cmd in "${DOWNLOADERS[@]}"; do
        echo "    $test_cmd"
    done
    exit 1
fi

ROOT_DIR=`dirname $0`
TGZ_FILE=`basename $TARGET`

UNPACK_DIR="$ROOT_DIR/tmp-getASL"
if test -e $UNPACK_DIR; then
    echo "Temporary directory ($UNPACK_DIR) exists!"
    echo "Cowardly refusing to overwrite."
    exit 1
fi
FINAL_DIR="$ROOT_DIR/solvers"
if test -e $FINAL_DIR; then
    echo "Final installation directory ($FINAL_DIR) exists!"
    echo "Cowardly refusing to overwrite."
    exit 1
fi

function fail() {
    MSG="$1"
    shift
    while test -n "$1"; do
        popd
        shift
    done
    rm -rf "$UNPACK_DIR"
    rm -rf "$FINAL_DIR"
    echo ""
    echo "$MSG"
    echo ""
    exit 1
}

mkdir "$UNPACK_DIR" || fail "Could not create temporary dir ($UNPACK_DIR)"
pushd "$UNPACK_DIR" || fail "Could not move to temporary dir ($UNPACK_DIR)"

echo "Downloading $TARGET"
$DOWNLOAD $TARGET
if test $? -eq 0; then
    echo "Download complete."
else
    fail "Download failed." 1
fi

tar -xzf $ASL.tar.gz || fail "Extracting archive failed" 1
mv */src/asl/solvers . || fail "Did not locate ASL solvers directory" 1
pushd solvers || fail "pushd failed"

echo "Updating CFLAGS"

mv makefile.u makefile.u.orig || fail "moving makefile failed" 2 1
sed -e 's/CFLAGS = /CFLAGS = -DNo_dtoa -fPIC /g' makefile.u.orig > makefile.u \
    || fail "Updating CFLAGS failed" 2 1

echo "Patching ASL"
patch < ../../asl.patch || fail "patching ASL failed" 2 1

popd || fail "popd failed" 2 1
popd || fail "popd failed" 1

mv "$UNPACK_DIR/solvers" "$FINAL_DIR" \
    || fail "Cound move ASL to final dir ($FINAL_DIR)"

echo "Deleting the temporary directory"
rm -rf "$UNPACK_DIR"

echo " "
echo "Done downloading the source code for ASL."
echo " "
