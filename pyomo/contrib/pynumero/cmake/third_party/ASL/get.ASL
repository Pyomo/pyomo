#!/bin/sh

set -e

wgetcmd="wget"
wgetcount=`which wget 2>/dev/null | wc -l`
if test ! $wgetcount = 1; then
  echo "Utility wget not found in your PATH."
  if test `uname` = Darwin; then
    wgetcmd="curl -L -k -O"
    echo "Using ftp command instead."
  elif test `uname` = FreeBSD; then
    wgetcmd=fetch
    echo "Using fetch command instead."
  else
    exit -1
  fi
fi

coinasl=1.3.0

echo " "
echo "Running script for downloading the source code for the ASL"
echo " "

rm -f $coinasl.tar.gz

echo "Downloading the source code from Github..."
if $wgetcmd https://github.com/ampl/mp/archive/$coinasl.tar.gz ;
then
  echo "Download finished."
else
  echo
  echo "Download failed...exiting"
fi

rm -rf solvers

echo "Unpacking the source code..."
gunzip -f $coinasl.tar.gz
tar xf $coinasl.tar
mv mp-$coinasl/src/asl/solvers .
rm -rf mp-$coinasl

echo "Adding No_dtoa to CFLAGS..."
cd solvers
mv makefile.u makefile.u.orig
sed -e 's/CFLAGS = /CFLAGS = -DNo_dtoa /g' makefile.u.orig > makefile.u 
cd ..

echo "Deleting the tar file..."
rm $coinasl.tar

echo " "
echo "Done downloading the source code for ASL."
echo " "

