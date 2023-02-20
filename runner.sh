#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$(readlink -f "${SCRIPT_DIR}")
export OPENBLAS_NUM_THREADS=1

/home/USER/envs/ENVS_NAME/bin/python $@
