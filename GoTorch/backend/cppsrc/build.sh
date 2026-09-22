#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

EXT_SUFFIX="$(${PYTHON_BIN} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
INCLUDES="$(${PYTHON_BIN} -m pybind11 --includes)"

g++ -O3 -Wall -shared -std=c++17 -fPIC \
    ${INCLUDES} \
    -I"${ROOT_DIR}/GoTorch/backend/cppsrc" \
    "${ROOT_DIR}/GoTorch/backend/cppsrc/ndarray.cpp" \
    "${ROOT_DIR}/GoTorch/backend/cppsrc/bindings.cpp" \
    -o "${ROOT_DIR}/GoTorch/backend/native_backend${EXT_SUFFIX}"

echo "Built ${ROOT_DIR}/GoTorch/backend/native_backend${EXT_SUFFIX}"
