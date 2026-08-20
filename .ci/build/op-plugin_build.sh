#!/bin/bash
# 用于 op-plugin 门禁编译任务

set -e
pytorch_version=$1
py_version=$2
pr_id=$3
WORKSPACE=$4
TARGET_BRANCH=$5
ARCH=$(uname -m)
export CCACHE_BASEDIR=${WORKSPACE}/CODE/build/pytorch

function compile_v210() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.1.0/torch-2.1.0%2Bcpu-cp310-cp310-linux_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.1.0+cpu-cp310-cp310-linux_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.1.0/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    fi
}

function compile_v251() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.5.1/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.5.1+cpu-cp310-cp310-linux_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.5.1/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    fi
}

function compile_v260() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.6.0/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.6.0/torch-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v271() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.7.1/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.7.1/torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v280() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.8.0/torch-2.8.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.8.0/torch-2.8.0-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.8.0-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v290() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.9.0/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.9.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.9.0/torch-2.9.0-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.9.0-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v291() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.9.1/torch-2.9.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.9.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.9.1/torch-2.9.1-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.9.1+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v2100() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.10.0/torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.10.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
        export AUDITWHEEL_PLAT=manylinux_2_28_x86_64
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.10.0/torch-2.10.0-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.10.0-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v2110() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.11.0/torch-2.11.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.11.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
        export AUDITWHEEL_PLAT=manylinux_2_28_x86_64
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.11.0/torch-2.11.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.11.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_v2120() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.12.0/torch-2.12.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.12.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
        export AUDITWHEEL_PLAT=manylinux_2_28_x86_64
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/"${pr_id}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.12.0/torch-2.12.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/"${pr_id}"/torch-2.12.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_master() {
    export OMP_NUM_THREADS=32
    nightly_version=`cat ${WORKSPACE}/tmp_pytorch/requirements.txt | grep torch== | cut -c 8-25`
    nightly_date=${nightly_version:10:8}
    if [ ${#nightly_date} -ne 8 ]; then
       echo "Unable to get torch nightly package with version ${nightly_version}."
       exit 1
    fi

    # nightly_date="20250215"
    # nightly_version="2.7.0.dev20250215"

    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "${nightly_version}"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/master/${nightly_date}/torch-${nightly_version}%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        if [ $? -ne 0 ]; then
            echo "Unable to get torch nightly package with version ${nightly_version}."
            exit 1
        fi
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-${nightly_version}+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
        export AUDITWHEEL_PLAT=manylinux_2_28_x86_64
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/master/${nightly_date}/torch-${nightly_version}%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
        if [ $? -ne 0 ]; then
            echo "Unable to get torch nightly package with version ${nightly_version}."
            exit 1
        fi
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-${nightly_version}+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function compile_master_test() {
    test_version=`cat ${WORKSPACE}/tmp_pytorch/requirements.txt | grep torch== | cut -c 8-24`
    test_version=${test_version:0:6}
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${test_version}/torch-${test_version}%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-${test_version}+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}"/../output/ https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${test_version}/torch-${test_version}%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
        pip"${py_version}" install "${WORKSPACE}"/../output/torch-${test_version}+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function main() {
    mkdir -p "${WORKSPACE}"/../output/"${pr_id}"
    if [ "${pytorch_version}" == "v2.1.0" ]; then
        compile_v210
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.1.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.1.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.1.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.1.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.5.1" ]; then
        compile_v251
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.5.1
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.5.1_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.5.1-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.5.1-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.6.0" ]; then
        compile_v260
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.6.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.6.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.6.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.6.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.7.1" ]; then
        compile_v271
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.7.1
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.7.1_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.7.1-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.7.1-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.8.0" ]; then
        compile_v280
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.8.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.8.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.8.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.8.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.9.0" ]; then
        compile_v290
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.9.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.9.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.9.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.9.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.9.1" ]; then
        compile_v291
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.9.1
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.9.1_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.9.1-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.9.1-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.10.0" ]; then
        compile_v2100
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.10.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.10.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.10.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.10.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.11.0" ]; then
        compile_v2110
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.11.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.11.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.11.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.11.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "v2.12.0" ]; then
        compile_v2120
        cd "${WORKSPACE}"/CODE
        if [ "${TARGET_BRANCH}" == "master" ]; then
            sh ci/build.sh --python=3.10 --pytorch=v2.12.0
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.12.0_${ARCH}.tar.gz ./
        else
            sh ci/build.sh --python=3.10 --pytorch=v2.12.0-"${TARGET_BRANCH}"
            tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_v2.12.0-"${TARGET_BRANCH}"_${ARCH}.tar.gz ./
        fi
    elif [ "${pytorch_version}" == "master" ]; then
        bash ${WORKSPACE}/cie/.ci/common/retry_command.sh "rm -rf ${WORKSPACE}/tmp_pytorch && git clone --depth=1 https://gitcode.com/Ascend/pytorch.git ${WORKSPACE}/tmp_pytorch"
        torch_version=`cat ${WORKSPACE}/tmp_pytorch/requirements.txt | grep torch== | cut -c 8-25`
        nightly_date=${torch_version:10:8}
        if [ ${#nightly_date} -ne 8 ]; then
            compile_master_test
        else
            compile_master
        fi
        cd "${WORKSPACE}"/CODE
        sh ci/build.sh --python=3.10 --pytorch=master
        tar -C dist -cvzf "${WORKSPACE}"/../output/"${pr_id}"/torch_npu_master_${ARCH}.tar.gz ./
    fi
    cp -rf "${WORKSPACE}"/modify_files.txt "${WORKSPACE}"/../output/"${pr_id}"
    cd "${WORKSPACE}"/../output/"${pr_id}"
    outcome=$(ls "${WORKSPACE}"/../output/"${pr_id}" | grep torch_npu)
    build_json $outcome
    cp -rf "${WORKSPACE}"/../output/"${pr_id}"/* "${WORKSPACE}"
#    obsutil cp "${WORKSPACE}"/../output/"${pr_id}" obs://pytorch-package/op-plugin/PR/ -r -f -acl=public-read

}

function build_json() {
    pro_name=$1
    url="https://pta-pr.obs.cn-north-4.myhuaweicloud.com/op-plugin/PR/${pr_id}/${pro_name}"
    python3.10 - << EOF
import json

data = {
    "type": "build",
    "buildings": [
        {
            "name": "$pro_name",
            "type": "二进制",
            "downloadUrl": "$url"
        }
    ]
}

with open("${pro_name}.json", "a", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
EOF
  echo "openLiBing build result JSON: ${url}.json"
}

if [ "${ARCH}" == "aarch64" ] && [ "${pytorch_version:0:6}" == "v2.6.0" ]; then
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "_GLIBCXX_USE_CXX11_ABI=1"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    export _GLIBCXX_USE_CXX11_ABI=1
else
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "_GLIBCXX_USE_CXX11_ABI=1"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    export _GLIBCXX_USE_CXX11_ABI=1
fi

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

if [ "${only_markdown}" == "True" ]; then
    echo "no need exec build"
    exit 0
fi

support_720="v2.1.0 v2.6.0 v2.7.1 v2.8.0"
support_730="v2.6.0 v2.7.1 v2.8.0 v2.9.0"
support_2600="v2.7.1 v2.8.0 v2.9.0 v2.10.0"
support_2610="v2.7.1 v2.9.0 v2.10.0 v2.11.0 v2.12.0"

suffix="${TARGET_BRANCH//./}"
list="support"_${suffix}

if [[ " $TARGET_BRANCH " != " master " ]] && [[ ! " ${!list} " =~ " $pytorch_version " ]]; then
    echo "$TARGET_BRANCH no need execute $pytorch_version build"
    wget --no-check-certificate --no-verbose -P "${WORKSPACE}"  https://pta-pr.obs.cn-north-4.myhuaweicloud.com/op-plugin/op.tar.gz
    exit 0
else
    echo "start execute build"
    main
    ccache -p -V -s
    echo "end execute build"
fi
