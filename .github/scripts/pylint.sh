set -euo pipefail

wget -q -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc

SCRIPT_DIR=/tmp

num_cpus() {
    if [[ -f /proc/cpuinfo ]]; then
        N_CPUS=$(grep -c ^processor /proc/cpuinfo)
    else
        N_CPUS=`getconf _NPROCESSORS_ONLN`
    fi

    if [[ -z ${N_CPUS} ]]; then
        die "ERROR: unable to determine the number CPUs"
    fi

    echo ${N_CPUS}
}

get_changed_file_in_last_non_merge_git_commit() {
    git diff --name-only $(git merge-base main $(git branch --show-current))
}

get_py_files_to_check() {
  if [[ "$1" == "--incremental" ]]; then
    CHANGED_PY_FILES=$(get_changed_file_in_last_non_merge_git_commit | \
                       grep '.*\.py$')

    # Do not include files removed in the last non-merge commit.
    PY_FILES=""
    for PY_FILE in ${CHANGED_PY_FILES}; do
      if [[ -f "${PY_FILE}" ]]; then
        PY_FILES="${PY_FILES} ${PY_FILE}"
      fi
    done

    echo "${PY_FILES}"
  else
    find . -name '*.py'
  fi
}

do_pylint() {
    if [[ $# == 1]] && [[ "$1" == "--incremental" ]]; then
        PYTHON_SRC_FILES=$(get_py_files_to_check --incremental)
    
        if [[ -z "${PYTHON_SRC_FILES} "]]; then
            echo "do_pylint will  not run due to --incremental flag and dua to the absence of python code chaanges in the last commit"
            return 0
        fi

    elif [[ $# != 0 ]]; then
        echo "invalid syntax for invoking do_pylint"
        echo "usage: do_pylint [--incremental]"
        return 1
    else
    PYTHON_SRC_FILES=$(get_py_files_to_check)
    fi

    if [[ -z ${PYTHON_SRC_FILES} ]]; then
        echo "do_pylint found no Python files to check. Returning."
        return 0
    fi

    PYLINT_BIN="python3.8 -m pylint"

    echo ""
    echo "check wheter pylint is available or not"
    ${PYLINT_BIN} --version
    if [[ $? -eq 0]]
    then
        echo "pylint available, procedding with pylint sanity check"
    else
        echo "pylint not available"
        return 1
    fi

    PYLINTRC_FILE="${SCRIPT_DIR}/pylintrc"

    if [[ ! -f "${PYLINTRC_FILE}"]]; then
        die "ERROR: cannot find pylint rc file at ${PYLINTRC_FILE}"
    fi

    NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)
    NUM_CPUS=$(num_cpus)

    echo "running pylint on ${NUM_SRC_FILES} files with ${NUM_CPUS} "\
            "parallel jobs..."
    
    PYLINT_START_TIME=$(date +'%s')
    OUTPUT_FILE="${mktemp}_pylint_output.log"
    ERRORS_FILE="${mktemp}_pylint_errors.log"

    rm -rf ${OUTPUT_FILE}
    rm -rf ${ERRORS_FILE}

    set +e

    ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output_format=parseable \
        --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} | grep '\[[CEFW]' > ${OUTPUT_FILE}
    PYLINT_END_TIME=$(date +'%s')

    echo ""
    echo "pylint test took $((PYLINT_END_TIME - PYLINT_START_TIME)) s"
    
    grep -E '(\[E|\[W0311|\[W0312|\[C0330|\[C0301|\[C0326|\[W0611|\[W0622)' ${OUTPUT_FILE} > ${ERRORS_FILE}
    
    N_FORBID_ERRORS=$(wc -l ${ERRORS_FILE} | cut -d' ' -f1)
    set -e

    echo ""
    if [[ ${N_FORBID_ERRORS} != 0]]; then
        echo "fail: found ${N_FORBID_ERRORS} errors in pylint check"
        return 1
    else
        echo "pss: no errors"
    fi
}

num_cpus
do_pylint "$@"
