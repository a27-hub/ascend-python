#!/bin/bash
# 用于codearts 预合并主仓代码
set -e
TARGET_BRANCH=$1
REMOTE_URL=$2
pr_id=$3
WORKSPACE=$4

cd ${WORKSPACE}
echo ${REMOTE_URL}
if [[ "${REMOTE_URL}" == *torchair* ]];then
  bash ${WORKSPACE}/cie/.ci/common/retry_command.sh "rm -rf torchair && git clone -b ${TARGET_BRANCH}  ${REMOTE_URL} torchair"
  cd ./torchair
else
  bash ${WORKSPACE}/cie/.ci/common/retry_command.sh "rm -rf CODE && git clone -b ${TARGET_BRANCH}  ${REMOTE_URL} CODE"
  cd ./CODE
fi
git config --global user.email "AtlasAccount@noreply.gitcode.com"
git config --global user.name "AtlasAccount"
if [ "${TRIGGER_TYPE}" == "MR" ];then
    echo "merged_commit_id: ${merged_commit_id}"
    bash ${WORKSPACE}/cie/.ci/common/retry_command.sh "git pull --depth=20 && git checkout ${merged_commit_id}"
    git checkout -b PR_MERGED_${TIMESTAMP}
else
    git fetch origin
    git fetch origin refs/merge-requests/${pr_id}/head:new_${pr_id}
    git checkout new_${pr_id}
    git merge origin/${TARGET_BRANCH}
fi
cd -