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

get_changed_file_in_last_non_merge_commit() {
    git diff --name-only $(git merge-base main $(git branch --show-current))
}

num_cpus()