#!/bin/sh

SCRIPT_DIR=$(dirname "$(realpath "$0")")

export FINN_ROOT=$(realpath "$SCRIPT_DIR/../finn")
export FINN_BUILD_DIR=${SCRIPT_DIR}/build

mkdir -p ${FINN_BUILD_DIR}

cd ${FINN_ROOT}
./run-docker.sh build_custom ${SCRIPT_DIR}

cd "${SCRIPT_DIR}/output_sfc_1w1a_fpga/sim"
vivado -mode batch -source make_sim_proj.tcl
cd "${SCRIPT_DIR}"

