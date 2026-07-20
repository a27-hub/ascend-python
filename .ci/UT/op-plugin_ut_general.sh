#!/bin/bash
# 用于op-plugin门禁UT 脚本
# shellcheck source=/dev/null

pytorch_version=$1
py_version=$2
pr_id=$3
WORKSPACE=$4
TARGET_BRANCH=$5
CUR_DIR=$(dirname $(readlink -f $0))
ARCH=$(uname -m)
source /etc/profile
export PTA_UT_EXEC_TIMEOUT=4800
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python_path=$(dirname $(dirname $(readlink -f $(which python$py_version))))

function judge_skip_ut() {
    # only modify markdown
    only_markdown="True"
    if [ -e ${WORKSPACE}/modify_files.txt ]; then
        while IFS= read -r line; do
            if [ "${line}" != "" ]; then
                line_suffix=${line: -3}
                if [ "${line_suffix}" != ".md" ] && [ "${line_suffix}" != 'md"' ]; then
                    only_markdown="False"
                fi
            fi
        done < ${WORKSPACE}/modify_files.txt
    else
        only_markdown="False"
    fi
    # only modify notice
    only_notice="True"
    if [ -e ${WORKSPACE}/modify_files.txt ]; then
        while IFS= read -r line; do
            if [ "${line}" != "" ]; then
                if [ "${line}" != "Third_Party_Open_Source_Software_Notice" ]; then
                    only_notice="False"
                fi
            fi
        done < ${WORKSPACE}/modify_files.txt
    else
        only_notice="False"
    fi

    if [ "${only_markdown}" == "True" ] || [ "${only_notice}" == "True" ]; then
        echo "no need exec UT"
        exit 0
    fi
}

function torch_install_other() {
    if [ "${ARCH}" == "x86_64" ]; then
        if [ "${1}" == "2.7.1" ] || [ "${1}" == "2.8.0" ] || [ "${1}" == "2.9.0" ] || [ "${1}" == "2.9.1" ] || [ "${1}" == "2.10.0" ] || [ "${1}" == "2.11.0" ] || [ "${1}" == "2.12.0" ]; then
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}%2Bcpu-cp${2}-cp${2}-manylinux_2_28_x86_64.whl
        else
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}%2Bcpu-cp${2}-cp${2}-linux_x86_64.whl
        fi
    elif [ "${ARCH}" == "aarch64" ]; then
        if [ "${1}" == "2.6.0" ] || [ "${1}" == "2.7.1" ] || [ "${1}" == "2.8.0" ] || [ "${1}" == "2.9.0" ] || [ "${1}" == "2.10.0" ]; then
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}-cp${2}-cp${2}-manylinux_2_28_aarch64.whl
        elif [ "${1}" == "2.9.1" ] || [ "${1}" == "2.11.0" ] || [ "${1}" == "2.12.0" ]; then
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}%2Bcpu-cp${2}-cp${2}-manylinux_2_28_aarch64.whl
        else
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}-cp${2}-cp${2}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        fi
    fi
}

function torch_install_master() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/master/${nightly_date}/torch-${nightly_version}%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/master/${nightly_date}/torch-${nightly_version}%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function torch_install_master_2() {
    test_version=${nightly_version:0:6}
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${test_version}/torch-${test_version}%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${test_version}/torch-${test_version}%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function torch_npu_install() {
    if [ "${TARGET_BRANCH}" == "master" ]; then
        tar_name=torch_npu_${pytorch_version}_${ARCH}.tar.gz
    else
        tar_name=torch_npu_${pytorch_version}-${TARGET_BRANCH}_${ARCH}.tar.gz
    fi
#    obsutil cp obs://pytorch-package/op-plugin/PR/${pr_id}/${tar_name} ./

    wget --no-host-directories -c  --no-check-certificate https://pta-pr.obs.cn-north-4.myhuaweicloud.com/op-plugin/PR/${pr_id}/${tar_name}
    tar -zxvf ${tar_name}
    pip"${py_version}" install -U "${WORKSPACE}"/torch-*${ARCH}.whl --force-reinstall
    pip"${py_version}" install -U "${WORKSPACE}"/torch_npu*${ARCH}.whl --force-reinstall --no-deps
}

function main() {
    judge_skip_ut
    #pip"${py_version}" install https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/triton-ascend/triton_ascend-3.4.0.dev2026030122-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
    pip"${py_version}" install https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/Triton_Innersource/triton-ascend/20260428190056/triton_ascend-3.2.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
    cd ${WORKSPACE}
    pip"${py_version}" uninstall -y torch
    pip"${py_version}" uninstall -y torch_npu

    # handle residual files of torch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/~orch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/-orch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch*.dist-info
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torchgen
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/~orchgen
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/-orchgen
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/functorch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/~unctorch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/-unctorch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch_npu
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch_npu*.dist-info
    if [ -d "/root/.cache/torch_extensions/py${py_version//./}_cpu/custom_add" ];then
        rm -rf /root/.cache/torch_extensions/py"${py_version//./}"_cpu/custom_add
    fi


    if [ ${pytorch_version:0:6} == "v2.1.0" ] || [ ${pytorch_version:0:6} == "v2.6.0" ] || [ ${pytorch_version:0:6} == "v2.7.1" ] || [ ${pytorch_version:0:6} == "v2.8.0" ] || [ ${pytorch_version:0:6} == "v2.9.0" ]; then
        torch_install_other "${pytorch_version:1:5}" "${py_version//./}"
    elif [ ${pytorch_version:0:7} == "v2.10.0" ] || [ ${pytorch_version:0:7} == "v2.11.0" ]|| [ ${pytorch_version:0:7} == "v2.12.0" ] ; then
        torch_install_other "${pytorch_version:1:6}" "${py_version//./}"
    elif [ "${pytorch_version}" = "master" ]; then
        bash ${WORKSPACE}/.ci/common/retry_command.sh "rm -rf ${WORKSPACE}/tmp_pytorch && git clone --depth=1 https://gitcode.com/Ascend/pytorch.git ${WORKSPACE}/tmp_pytorch"
        nightly_version=`cat ${WORKSPACE}/tmp_pytorch/requirements.txt | grep torch== | cut -c 8-25`
        nightly_date=${nightly_version:10:8}
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "${nightly_version}"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        if [ ${#nightly_date} -ne 8 ]; then
            torch_install_master_2
        else
            torch_install_master
        fi
        pip"${py_version}" install -r ${WORKSPACE}/tmp_pytorch/test/requirements.txt
#        wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torchvision/master/torchvision-0.27.0.dev20260424%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
#        pip"${py_version}" install -U "${WORKSPACE}"/torchvision-*.whl --force-reinstall --no-deps
    fi
    torch_npu_install
    pip${py_version} install einops
    pip${py_version} install tlparse==0.3.18
    pip${py_version} install tabulate==0.9.0
    pip${py_version} list

    cp -rf modify_files.txt ./CODE
    if [ "${pytorch_version}" != "master" ]; then
        cp -f ${WORKSPACE}/.ci/UT/op-plugin/common_files.txt ./CODE
    fi
    cd ${WORKSPACE}/CODE
    rm -rf ci/exec_ut.sh
    cp -f ${WORKSPACE}/.ci/UT/op-plugin/exec_ut.sh ${WORKSPACE}/CODE/ci/exec_ut.sh
    cp -f ${WORKSPACE}/.ci/UT/op-plugin/delete_torchair_base_schema.py ${WORKSPACE}/CODE/ci/delete_torchair_base_schema.py

    pkill -f /usr/local/bin/python3 || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f /opt/_internal/cpython || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f "${python_path}"/bin/python"${py_version}" || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f /usr/local/bin/python"${py_version}" || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f python"${py_version}" || echo "IGNORE KILL ERROR" > /dev/null
    ps -ef | grep "/usr/local/bin/python3" | grep -v grep | awk '{print $2}' | xargs kill -9 2> /dev/null || echo "IGNORE KILL ERROR" > /dev/null
    ps -ef | grep "/opt/_internal/cpython" | grep -v grep | awk '{print $2}' | xargs kill -9 2> /dev/null || echo "IGNORE KILL ERROR" > /dev/null

    export PATH="${python_path}"/bin:$PATH
    python"${py_version}" -c "import acl;print('npu info:',acl.get_soc_name())" || echo "npu info  get error"

    echo "------------------------------------- ENV PATH -------------------------------------"
    echo ${PATH}
    echo "------------------------------------------------------------------------------------"

    if [ "${TARGET_BRANCH}" == "master" ]; then
        sh ci/exec_ut.sh --python=${py_version} --pytorch=${pytorch_version} --pr_id=${pr_id}
    else
        sh ci/exec_ut.sh --python=${py_version} --pytorch=${pytorch_version}-${TARGET_BRANCH} --pr_id=${pr_id}
    fi
}

set -e

support_730="v2.6.0 v2.7.1 v2.8.0 v2.9.0"
support_720="v2.1.0 v2.6.0 v2.7.1 v2.8.0"
support_2600="v2.7.1 v2.8.0 v2.9.0 v2.10.0"
support_2610="v2.7.1 v2.9.0 v2.10.0 v2.11.0 v2.12.0"

suffix="${TARGET_BRANCH//./}"
list="support"_${suffix}

if [[ " $TARGET_BRANCH " != " master " ]] && [[ ! " ${!list} " =~ " $pytorch_version " ]]; then
    echo "$TARGET_BRANCH no need execute $pytorch_version ut"
    exit 0
else
    echo "start execute ut"
    main
    echo "end execute ut"
fi
