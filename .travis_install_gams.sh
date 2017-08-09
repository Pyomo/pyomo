#
# Install free version of GAMS: executable and API bindings
# No explicit API version for python 3.5, use 3.4
#
wget https://d37drm4t2jghv5.cloudfront.net/distributions/24.8.5/linux/linux_x64_64_sfx.exe
chmod u+x linux_x64_64_sfx.exe
./linux_x64_64_sfx.exe > /dev/null
PATH=${PATH}:$PWD/gams24.8_linux_x64_64_sfx
# Change default NLP solver to baron for testing, performs better than CONOPT
pushd $PWD/gams24.8_linux_x64_64_sfx
sed -i 's/NLP CONOPT/NLP BARON/g' gmscmpun.txt
popd
if [ "${TRAVIS_PYTHON_VERSION}" = "2.6" ]; then
   pushd $PWD/gams24.8_linux_x64_64_sfx/apifiles/Python/api_26;
   python setup.py install;
   popd;
fi
if [ "${TRAVIS_PYTHON_VERSION}" = "2.7" ]; then
   pushd $PWD/gams24.8_linux_x64_64_sfx/apifiles/Python/api;
   python setup.py install;
   popd;
fi
if [ "${TRAVIS_PYTHON_VERSION}" = "3.4" ]; then
   pushd $PWD/gams24.8_linux_x64_64_sfx/apifiles/Python/api_34;
   python setup.py install;
   popd;
fi
if [ "${TRAVIS_PYTHON_VERSION}" = "3.5" ]; then
   pushd $PWD/gams24.8_linux_x64_64_sfx/apifiles/Python/api_34;
   python setup.py install;
   popd;
fi
if [ "${TRAVIS_PYTHON_VERSION}" = "3.6" ]; then
   pushd $PWD/gams24.8_linux_x64_64_sfx/apifiles/Python/api_36;
   python setup.py install;
   popd;
fi
