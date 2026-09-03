#!/bin/bash
# 用于codearts 预合并主仓代码
set -e
TARGET_BRANCH=$1
REMOTE_URL=$2
pr_id=$3
WORKSPACE=$4

cd ${WORKSPACE}
echo ${REMOTE_URL}
bash ${WORKSPACE}/cie/.gitcode/workflows/scripts/common/retry_command.sh "rm -rf CODE && git clone -b ${TARGET_BRANCH}  ${REMOTE_URL} CODE"
cd ./CODE
git config --global user.email "AtlasAccount@noreply.gitcode.com"
git config --global user.name "AtlasAccount"
git fetch origin
git fetch origin refs/merge-requests/${pr_id}/head:new_${pr_id}
git checkout new_${pr_id}
git merge origin/${TARGET_BRANCH}
