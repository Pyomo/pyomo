#
# Install GAMS-Python API bindings
# No explicit API version for python 3.5, use 3.4
#
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
