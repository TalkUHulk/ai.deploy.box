#!/bin/bash
#source set_env.sh

CURRENT_DIR=$(dirname $(realpath ${BASH_SOURCE}))
lib_path="${CURRENT_DIR}/libs/linux/x86_64"
lib_tnn_path="${CURRENT_DIR}/libs/linux/x86_64/tnn"
lib_openvino_path="${CURRENT_DIR}/libs/linux/x86_64/openvino"
lib_paddlelite_path="${CURRENT_DIR}/libs/linux/x86_64/paddlelite"


LD_LIBRARY_PATH=${lib_path}:${lib_tnn_path}:${lib_openvino_path}:${lib_paddlelite_path}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH


