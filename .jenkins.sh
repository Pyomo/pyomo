#!/usr/bin/env bash
# Important environment variables:
#
# WORKSPACE: path to the base WORKSPACE.  This script assumes that there
#     are 2 available subdirectories: pyomo (the pyomo source checkout
#     and pyomo-model-libraries
#     (the checkout of the additional model libraries repo).  It will
#     create two additional directories within WORKSPACE: python (a
#     virtualenv) and config (the local Pyomo configuration/cache
#     directory)
#
# CATEGORY: the category to pass to pytest
#
# TEST_SUITES: Paths (module or directory) to be passed to pytest to
#     run. (defaults to "pyomo '$WORKSPACE/pyomo-model-libraries'")
#
# SLIM: If nonempty, then the virtualenv will only have pip, setuptools,
#     and wheel installed.  Otherwise the virtualenv will inherit the
#     system site-packages.
#
# CODECOV_TOKEN: the token to use when uploading results to codecov.io
#
# CODECOV_SOURCE_BRANCH: passed to the 'codecov-cli' command; branch of Pyomo
#     (e.g., to enable correct codecov uploads)
#
# CODECOV_REPO_OWNER: passed to the 'codecov-cli' command; owner of repo
#     (e.g., to enable correct codecov uploads)
#
# DISABLE_COVERAGE: if nonempty, then coverage analysis is disabled
#
# PYOMO_SETUP_ARGS: passed to the 'python setup.py develop' command
#     (e.g., to specify --with-cython)
#
# PYOMO_DOWNLOAD_ARGS: passed to the 'pyomo download-extensions' command
#     (e.g., to set up local SSL certificate authorities)
#
# PYTEST_EXTRA_ARGS: passed to the 'pytest' command
#     (e.g., to add extra pytest options like '--collect-only')
#
if test -z "$WORKSPACE"; then
    export WORKSPACE=`pwd`
fi
if test -z "$TEST_SUITES"; then
    export TEST_SUITES="${WORKSPACE}/pyomo/pyomo ${WORKSPACE}/pyomo-model-libraries ${WORKSPACE}/pyomo/examples ${WORKSPACE}/pyomo/doc"
fi
if test -z "$SLIM"; then
    export VENV_SYSTEM_PACKAGES='--system-site-packages'
fi

if test "$WORKSPACE" != "`pwd`"; then
    echo "ERROR: pwd is not WORKSPACE"
    echo "   pwd=       `pwd`"
    echo "   WORKSPACE= $WORKSPACE"
    exit 1
fi
MODE="$1"

if test -z "$MODE" -o "$MODE" == setup; then
    # Clean old PYC files and remove any previous virtualenv
    echo "#"
    echo "# Removing python virtual environment"
    echo "#"
    rm -rf ${WORKSPACE}/python
    echo "#"
    echo "# Cleaning out old .pyc and cython files"
    echo "#"
    for EXT in pyc pyx pyd so dylib dll; do
        find ${WORKSPACE}/pyomo -name \*.$EXT -delete
    done

    # Set up the local lpython
    echo ""
    echo "#"
    echo "# Setting up virtual environment"
    echo "#"
    virtualenv python $VENV_SYSTEM_PACKAGES --clear || exit 1
    source python/bin/activate
    # Because modules set the PYTHONPATH, we need to make sure that the
    # virtualenv appears first
    LOCAL_SITE_PACKAGES=`python -c "import sysconfig; print(sysconfig.get_path('purelib'))"`
    export PYTHONPATH="$LOCAL_SITE_PACKAGES:$PYTHONPATH"

    # Set up Pyomo checkouts
    echo ""
    # configure the Pyomo configuration directory
    echo "#"
    echo "# Installing pyomo modules"
    echo "#"
    if test -d "$WORKSPACE/pyutilib"; then
        pushd "$WORKSPACE/pyutilib"
        python setup.py develop || echo "PyUtilib failed - skipping."
        popd
    else
        echo "PyUtilib not found; skipping"
    fi
    pushd "$WORKSPACE/pyomo" || exit 1
    python setup.py develop $PYOMO_SETUP_ARGS || exit 1
    popd
    #
    # DO NOT install pyomo-model-libraries
    #

    # Set up coverage tracking for subprocesses
    if test -z "$DISABLE_COVERAGE"; then
        # Clean up old coverage files
        rm -fv ${WORKSPACE}/pyomo/.coverage ${WORKSPACE}/pyomo/.coverage.*
        # Set up coverage for this build
        export COVERAGE_PROCESS_START=${WORKSPACE}/coveragerc
        cp ${WORKSPACE}/pyomo/.coveragerc ${COVERAGE_PROCESS_START}
        echo "data_file=${WORKSPACE}/pyomo/.coverage" \
            >> ${COVERAGE_PROCESS_START}
        echo 'import coverage; coverage.process_startup()' \
            > "${LOCAL_SITE_PACKAGES}/run_coverage_at_startup.pth"
    fi

    # Move into the pyomo directory
    pushd ${WORKSPACE}/pyomo || exit 1

    # Set a local Pyomo configuration dir within this workspace
    export PYOMO_CONFIG_DIR="${WORKSPACE}/config"
    echo ""
    echo "PYOMO_CONFIG_DIR=$PYOMO_CONFIG_DIR"
    echo ""

    # Call Pyomo build scripts to build TPLs that would normally be
    # skipped by the pyomo download-extensions / build-extensions
    # actions below
    if [[ " $CATEGORY " == *" builders "* ]]; then
        echo ""
        echo "Running local build scripts..."
        echo ""
        set -x
        python pyomo/contrib/simplification/build.py --build-deps || exit 1
        set +x
    fi

    # Use Pyomo to download & compile binary extensions
    i=0
    while /bin/true; do
        i=$[$i+1]
        echo ""
        echo "Downloading pyomo extensions (attempt $i)"
        pyomo download-extensions $PYOMO_DOWNLOAD_ARGS
        if test $? == 0; then
            break
        elif test $i -ge 3; then
            exit 1
        fi
        DELAY=$(( RANDOM % 30 + 15))
        echo "Pausing $DELAY seconds before re-attempting download"
        sleep $DELAY
    done
    pyomo build-extensions || exit 1

    # Print useful version information
    echo ""
    echo "#"
    echo "# Package information:"
    echo "#"
    python --version
    pip --version
    pip list
    echo "#"
    echo "# Installed programs:"
    echo "#"
    gjh -v || echo "GJH not found"
    glpsol -v || echo "GLPK not found"
    cbc -quit || echo "CBC not found"
    cplex -c quit || echo "CPLEX not found"
    gurobi_cl --version || echo "GUROBI not found"
    ipopt -v || echo "IPOPT not found"
    gams || echo "GAMS not found"

    # Exit ${WORKSPACE}/pyomo
    popd
fi

if test -z "$MODE" -o "$MODE" == test; then
    # Copy conftest.py into every requested test suite that is NOT
    # within ${WORKSPACE}/pyomo
    for TEST in $TEST_SUITES; do
      if [[ "$TEST" != *"${WORKSPACE}/pyomo/"* ]]; then
        cp ${WORKSPACE}/conftest.py $TEST
      fi;
    done
    rm ${WORKSPACE}/conftest.py
    echo ""
    echo "#"
    echo "# Running Pyomo tests"
    echo "#"
    python -m pytest -v \
        -W ignore::Warning \
        --junitxml="TEST-pyomo.xml" \
        -m "$CATEGORY" $TEST_SUITES $PYTEST_EXTRA_ARGS

    # Combine the coverage results and upload
    if test -z "$DISABLE_COVERAGE"; then
        # Enter ${WORKSPACE}/pyomo for coverage Processing
        pushd ${WORKSPACE}/pyomo || exit 1
        echo ""
        echo "#"
        echo "# Processing coverage information in "`pwd`
        echo "#"
        #
        # Note, that the PWD should still be $WORKSPACE/pyomo
        #
        coverage combine || exit 1
        coverage report -i || exit 1
        coverage xml -i || exit 1
        export OS=`uname`
        if test -z "$PYOMO_SOURCE_SHA"; then
            PYOMO_SOURCE_SHA=$GIT_COMMIT
        fi
        if test -n "$CODECOV_TOKEN" -a -n "$PYOMO_SOURCE_SHA"; then
            CODECOV_JOB_NAME=$(echo ${JOB_NAME} \
                | sed -r 's/^(.*autotest_)?Pyomo_([^\/]+).*/\2/').$BUILD_NUMBER.$python
            if test -z "$CODECOV_REPO_OWNER"; then
                if test -n "$PYOMO_SOURCE_REPO"; then
                    CODECOV_REPO_OWNER=$(echo "$PYOMO_SOURCE_REPO" | cut -d '/' -f 4)
                elif test -n "$GIT_URL"; then
                    CODECOV_REPO_OWNER=$(echo "$GIT_URL" | cut -d '/' -f 4)
                else
                    CODECOV_REPO_OWNER=""
                fi
            fi
            if test -z "$CODECOV_SOURCE_BRANCH"; then
                CODECOV_SOURCE_BRANCH=$(git branch -av --contains "$PYOMO_SOURCE_SHA" \
                    | grep "${PYOMO_SOURCE_SHA:0:7}" | grep "/origin/" \
                    | cut -d '/' -f 3 | cut -d' ' -f 1)
                if test -z "$CODECOV_SOURCE_BRANCH"; then
                    CODECOV_SOURCE_BRANCH=main
                fi
            fi
            i=0
            while /bin/true; do
                i=$[$i+1]
                echo "Uploading coverage to codecov (attempt $i)"
                codecovcli -v upload-process --sha $PYOMO_SOURCE_SHA \
                    --fail-on-error --git-service github --token $CODECOV_TOKEN \
                    --slug pyomo/pyomo --file coverage.xml --disable-search \
                    --name $CODECOV_JOB_NAME \
                    --branch $CODECOV_REPO_OWNER:$CODECOV_SOURCE_BRANCH \
                    --env OS,python --network-root-folder `pwd` --plugin noop
                if test $? == 0; then
                    break
                elif test $i -ge 4; then
                    exit 1
                fi
                DELAY=$(( RANDOM % 30 + 15))
                echo "Pausing $DELAY seconds before re-attempting upload"
                sleep $DELAY
            done
        fi
        rm .coverage
        # Exit ${WORKSPACE}/pyomo
        popd
    fi
fi
