#!/usr/bin/env python
# -*- coding: utf-8 -*-
# anti-poison CI script
import datetime
import sys
import time
import argparse
import requests
import json
from apig_sdk import signer


class AntiPoison(object):
    def __init__(self, created_appId, created_secretKey, result_appId, result_secretKey, org_name, pr_id,
                 repo, project_name,
                 repo_type='gitcode',
                 codecheck_ip='https://apig.openlibing.com',
                 gitxx_api='https://api.gitcode.com/api/v5/repos',
                 codecheck_prefix='/openlibing-anti-poison'):
        self.pr_url = None
        self.scanId = None
        self.result = None
        self.report_url = None
        self.comment = None
        self.timestamp = None
        self.sign = None
        self.created_appId = created_appId
        self.created_secretKey = created_secretKey
        self.result_appId = result_appId
        self.result_secretKey = result_secretKey
        self.codecheck_ip = codecheck_ip
        self.codecheck_prefix = codecheck_prefix
        self.project_name = project_name

        self.pr_id = pr_id
        self.org_name = org_name
        self.repo = repo
        self.gitxx_api = gitxx_api
        self.gitxx_domain_url = "https://gitcode.com"
        self.repo_type = repo_type

        self._set_pr_url()

    def _set_pr_url(self):
        self.pr_url = f"{self.gitxx_domain_url}/{self.org_name}/{self.repo}/pull/{self.pr_id}"
        print(f"pr_url: {self.pr_url}")

    def _is_json_check(self, text):
        try:
            json.loads(text)
        except ValueError:
            return False
        return True

    def _get_create_task(self):
        try:
            task_url = f'{self.codecheck_ip}{self.codecheck_prefix}/poison-pr/sca-pr'

            headers = {'Content-Type': 'application/json'}
            params = {
                "projectName": f"{self.project_name}",
                "prUrl": self.pr_url
            }
            method = 'POST'
            request = signer.HttpRequest(method, task_url, headers, json.dumps(params))
            sig = signer.Signer()
            sig.Key = self.created_appId
            sig.Secret = self.created_secretKey
            sig.Sign(request)
            response1 = requests.request(request.method, request.scheme + "://" + request.host + request.uri,
                                         headers=request.headers, data=request.body)

            if not self._is_json_check(response1.text):
                print(
                    f'Error: create tasak - response is not a valid json, status_code: {response1.status_code}, response.text: {response1.text}')
                sys.exit(-1)

            response = response1.json()
            result = response.get('result')
            if response.get('code') == 200:
                self.scanId = result.get('scanId')
                print('scanId:', self.scanId)
                print(f'[anti poison {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] create task success')
            else:
                self.comment = "task failed"
                print(
                    f'[anti poison {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] task failed, error code: {response.get("code")}, response.text: {response1.text}')
        except Exception as e:
            print(
                f'[anti poison {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] create task failed, error: {e}')

    def _get_task_result(self):
        print('get result')
        expire_time = 0
        while expire_time < 600:
            time.sleep(90)
            try:
                status_url = f'{self.codecheck_ip}{self.codecheck_prefix}/poison-pr/query-pr-results-status'

                headers = {'Content-Type': 'application/json'}
                params = {
                    "scanId": self.scanId
                }
                method = 'POST'
                request = signer.HttpRequest(method, status_url, headers, json.dumps(params))
                sig = signer.Signer()
                sig.Key = self.result_appId
                sig.Secret = self.result_secretKey
                sig.Sign(request)
                response1 = requests.request(request.method, request.scheme + "://" + request.host + request.uri,
                                             headers=request.headers, data=request.body)

                if not self._is_json_check(response1.text):
                    print(
                        f'Error: execute task - response is not a valid json, status_code: {response1.status_code}, response.text: {response1.text}')
                    sys.exit(-1)

                response = response1.json()
                result = response.get('result')
                if response.get('code') == 200:
                    self.result = result.get('isPass')
                    self.report_url = result.get('url')
                    self.comment = "pass"
                    if self.report_url is not None:
                        print(f'[anti poison {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] scan success')
                        break
                else:
                    self.comment = "task failed"
                    print(
                        f'[anti poison {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] task failed, error code: {response.get("code")}, response.text: {response1.text}')
                    break
            except Exception as e:
                print(
                    f'[anti poison {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] task failed exception, error: {e}')
                self.comment = "task failed"
                break

    def run(self):
        if not self.pr_url:
            sys.exit(-1)

        self._get_create_task()
        if not self.scanId:
            self.result = 'failed'
            print(f'openlibing_url:')
            sys.exit(-1)

        self._get_task_result()
        if not self.report_url or self.result is None:
            print(
                f'[antipoison check {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] get task results failed, timeout after multiple attempts.')
            sys.exit(-1)

        print(f'openlibing_url:{self.report_url}')
        if self.comment != "pass":
            print(f'[code sca {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] no pass')
            sys.exit(-1)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--repo_type', help='code hosting platform', default='gitcode', type=str)
    parser.add_argument('--org_name', help='organization name', default='Ascend', type=str)
    parser.add_argument('--pr_id', help='PR id', required=True, type=str)
    parser.add_argument('--repo', help='target repo', required=True, type=str)
    parser.add_argument('--gitxx_api', help='gitxx api', default='https://api.gitcode.com/api/v5/repos', type=str)

    parser.add_argument('--created_appId', help='account id', required=True, type=str)
    parser.add_argument('--created_secretKey', help='secret key', required=True, type=str)
    parser.add_argument('--result_appId', help='account id', required=True, type=str)
    parser.add_argument('--result_secretKey', help='secret key', required=True, type=str)
    parser.add_argument('--codecheck_ip', help='codesca ip', default='https://apig.openlibing.com', type=str)
    parser.add_argument('--codecheck_prefix', help='sca prefix', default='/openlibing-anti-poison', type=str)
    parser.add_argument('--project_name', help='belonging project name that in openlibing code repository service',
                        default='FrameworkPTAdapter', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()

    antiPoison = AntiPoison(args.created_appId, args.created_secretKey, args.result_appId, args.result_secretKey,
                            args.org_name,
                            args.pr_id, args.repo,
                            args.project_name)
    antiPoison.run()
