#!/bin/bash

# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

ARCH=$(uname -m)
CUR_DIR=$(dirname $(readlink -f $0))
PY_VERSION='3.8' # Default supported python version is 3.8
PYTORCH_VERSION='master' # Default supported PyTorch version is master
DEFAULT_SCRIPT_ARGS_NUM_MAX=3 # Default max supported input parameters
PR_ID='1'
export LD_PRELOAD=$LD_PRELOAD:/lib64/libgomp.so.1

# Parse arguments inside script
function parse_script_args() {
    local args_num=0

    while true; do
        if [[ "x${1}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${1}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ "x${2}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${2}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ "x${3}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${3}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ ${args_num} -eq ${DEFAULT_SCRIPT_ARGS_NUM_MAX} ]]; then
            break
        fi
        
    done

    while true; do
        case "${1}" in
        --python=*)
            PY_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --pytorch=*)
            PYTORCH_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --pr_id=*)
            PR_ID=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        -*)
            echo "ERROR Unsupported parameters: ${1}"
            return 1
            ;;
        *)
            if [ "x${1}" != "x" ]; then
                echo "ERROR Unsupported parameters: ${1}"
                return 1
            fi
            break
            ;;
        esac
    done

    # if some "--param=value" are not parsed correctly, throw an error.
    if [[ ${args_num} -ne 0 ]]; then
        return 1
    fi
}

function main()
{
    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi

    cd ${CUR_DIR}
    python"${PY_VERSION}" access_control_test.py --rank=${TEST_RANK} --world_size=${TEST_WORLD_SIZE}

    # clone torch_adapter for ops ut
    PYTORCH_PATH=${CUR_DIR}/../pytorch_ut
    if [ ! -d ${PYTORCH_PATH} ]; then
        bash "${WORKSPACE}"/cie/.ci/common/retry_command.sh "git clone -b ${PYTORCH_VERSION} https://gitcode.com/Ascend/pytorch.git ${PYTORCH_PATH}"
        cd ${PYTORCH_PATH}
        bash "${WORKSPACE}"/cie/.ci/common/retry_command.sh "git submodule update --init --recursive --force third_party/op-plugin  third_party/torchair/torchair"
        if [ -f "${WORKSPACE}/CODE/test/allowlist_for_publicAPI.json" ];then
           cp -f ${WORKSPACE}/CODE/test/allowlist_for_publicAPI.json ./third_party/op-plugin/test/allowlist_for_publicAPI.json
        fi
        cd -
        if [[ "${PYTORCH_VERSION}" =~ "6.0.rc" ]]; then
           echo "skip"
        elif [ "${PYTORCH_VERSION}" \> "v2.0.1" ] || [ "${PYTORCH_VERSION}" == "master" ]; then
            python"${PY_VERSION}" delete_torchair_base_schema.py -s "${PYTORCH_PATH}"/test/torch_npu_schema.json
            if [ -f "${PYTORCH_PATH}/third_party/torchair/torchair/tests/st/torch_npu_schema.json" ];then
               python"${PY_VERSION}" delete_torchair_base_schema.py -s "${PYTORCH_PATH}"/third_party/torchair/torchair/tests/st/torch_npu_schema.json
            fi
        fi
    fi

    # copy modify_files.txt to torch_adapter/ci
    cp ${CUR_DIR}/../modify_files.txt ${PYTORCH_PATH}/
    echo "copy common_files.txt for not master branch!"
    if [ "${PYTORCH_VERSION:0:6}" != "master" ]; then
        cp ${CUR_DIR}/../common_files.txt ${PYTORCH_PATH}/
    fi

    # obtain common test files from obs (sync with github)
    cd "${PYTORCH_PATH}"
    echo "sync with github"
    branch_path=${PYTORCH_VERSION%%-*}
    branch_path=${branch_path%%_*}
    echo "branch:${branch_path}"
    temp_folder="testcase_temp"
    rm -rf "${temp_folder}"
    mkdir -p "${temp_folder}"

    wget --no-host-directories -c -q --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/test/common_testcase/${branch_path}/testcase.tar.gz -O "${temp_folder}/testcase.tar.gz"

    tar -zxf "${temp_folder}/testcase.tar.gz" -C "${temp_folder}"

    if [ -e "${PYTORCH_PATH}"/test/testfiles_synchronized.txt ]; then
        cd "${temp_folder}"
        cat "${PYTORCH_PATH}"/test/testfiles_synchronized.txt | while IFS= read -r line; do
            line=$(echo "${line}" | tr -d '\r')
            if [ "${line}" != "" ]; then
                mkdir -p "${PYTORCH_PATH}"/"$(dirname "${line}")"
                cp -rf "${line}" "${PYTORCH_PATH}"/"${line}" || echo "testfiles copy failed: ${line}"
            fi
        done
        cd "${PYTORCH_PATH}"
    fi
    if [ -e "${PYTORCH_PATH}"/test/testfolder_synchronized.txt ]; then
        cd "${temp_folder}"
        cat "${PYTORCH_PATH}"/test/testfolder_synchronized.txt | while IFS= read -r line; do
            line=$(echo "${line}" | tr -d '\r')
            if [ "${line}" != "" ]; then
                cp -rf "${line}" "${PYTORCH_PATH}"/"${line}" || echo "testfolder copy failed: ${line}"
            fi
        done
        cd "${PYTORCH_PATH}"
    fi

    rm -rf "${temp_folder}"

    echo "Skip several test cases due to bug of py39."
    rm -rf ${PYTORCH_PATH}/test/test_jit.py
    rm -rf ${PYTORCH_PATH}/test/test_jit*.py
    rm -rf ${PYTORCH_PATH}/test/jit
    sed -i "s/'test_jit_fuser_te',/ /g" ${PYTORCH_PATH}/ci/access_control/constants.py
#        if [ ${PYTORCH_VERSION} = "master" ] || [ "${PYTORCH_VERSION:0:6}" == "v2.6.0" ];then
#            rm -rf ${PYTORCH_PATH}/test/distributed/test_register_sharding.py
#        fi
    rm -rf ${PYTORCH_PATH}/test/distributed/test_register_sharding.py
    rm -rf ${PYTORCH_PATH}/test/distributed/fsdp2
    if [ "${ARCH}" == "aarch64" ]; then
        rm -rf ${PYTORCH_PATH}/test/trans_contiguous
    fi

#    echo "Skip test_to_non_blocking in test_npu. [not stable]"
#    sed -i 's/_test_to_non_blocking(src, try_non_blocking, dst)/ /g' ${PYTORCH_PATH}/test/npu/test_npu.py


    # exec ut
    if [ "${PYTORCH_VERSION}" \> "v2.0.1" ] || [ "${PYTORCH_VERSION}" == "master" ]; then
        export DISABLED_TESTS_FILE=${PYTORCH_PATH}/test/unsupported_test_cases/.pytorch-disabled-tests.json
    fi
    # no torchair in op-plugin access_control
    rm -rf ${PYTORCH_PATH}/test/dynamo/*
    rm -rf ${PYTORCH_PATH}/test/_inductor/*

    if [ "${PY_VERSION}" == "3.7" ]; then
        export PATH=/opt/_internal/cpython-3.7.5/bin:$PATH
    elif [ "${PY_VERSION}" == "3.8" ]; then
        if [ "${ARCH}" == "x86_64" ]; then
            export PATH=/opt/_internal/cpython-3.8.1/bin:$PATH
        else
            export PATH=/opt/_internal/cpython-3.8.19/bin:$PATH
        fi
    elif [ "${PY_VERSION}" == "3.9" ]; then
        if [ "${ARCH}" == "x86_64" ]; then
            export PATH=/opt/_internal/cpython-3.9.21/bin:$PATH
        else
            export PATH=/opt/_internal/cpython-3.9.19/bin:$PATH
        fi
    elif [ "${PY_VERSION}" == "3.10" ]; then
        if [ "${ARCH}" == "x86_64" ]; then
            export PATH=/opt/_internal/cpython-3.10.16/bin:$PATH
        else
            export PATH=/opt/_internal/cpython-3.10.16/bin:$PATH
        fi
    fi
    cd ${PYTORCH_PATH}/ci
    echo "Start to run access_control_test in PTA."
    if [ "${PYTORCH_VERSION:0:6}" == "v2.6.0" ]; then
        cp -f ${WORKSPACE}/cie/.ci/UT/pta/test_manager.py ${PYTORCH_PATH}/ci/access_control/test_manager.py
    fi
    # export ASCEND_LAUNCH_BLOCKING=0
    python"${PY_VERSION}" access_control_test.py --rank=${TEST_RANK} --world_size=${TEST_WORLD_SIZE}

    exit 0
}

main "$@"
