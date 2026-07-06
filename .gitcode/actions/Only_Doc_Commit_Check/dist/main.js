/**
 * Only_Doc_Commit_Check Action
 *
 * 检查 PR 是否仅包含文档变更。
 * 调用 only_doc.py 脚本执行以下流程：
 *   1. clone 目标仓库，模拟 pre-merge，生成变更文件列表
 *   2. 判断所有变更文件是否均为文档类型
 *   3. 输出 JSON 结果 {"is_only_doc": "yes"/"no"}
 *
 */

const core = require("@actions/core");
const { execFile } = require("child_process");
const path = require("path");
const fs = require("fs").promises;
const fsSync = require("fs");

function info(message) {
    process.stdout.write(`${message}\n`);
}

function error(message) {
    process.stderr.write(`\x1b[31m[ERROR] ${message}\x1b[0m\n`);
}

function setFailed(message) {
    error(message);
    process.exit(1);
}

/**
 * 设置 action 输出变量
 * GitCode/AtomGit 工作流命令格式: ::set-output var=<key>:<value>
 * 参考 @actions/core setOutput: ::set-output name=<name>::<value>
 */
function setOutput(key, value) {
    core.setOutput(key, value);
}

// ============================================================
// Shell 脚本执行
// ============================================================

/**
 * 执行 bash 脚本
 * @param {string} script - 脚本路径
 * @param {string[]} args - 传入脚本的参数
 * @param {string} cwd - 工作目录
 * @param {number} timeoutMs - 超时时间（毫秒）
 * @returns {Promise<void>}
 */
function execShellScript(script, args, cwd, timeoutMs) {
    return new Promise((resolve, reject) => {
        const child = execFile("bash", [script, ...args], {
            cwd,
            timeout: timeoutMs,
            env: {
                ...process.env,
                SCRIPT_DIR: path.dirname(script),
            },
        });
        child.stdout?.on("data", (data) => {
            info(data.toString().trimEnd());
        });
        child.stderr?.on("data", (data) => {
            info(data.toString().trimEnd());
        });
        child.on("close", (code) => {
            if (code === 0) resolve();
            else if (code === null) reject(new Error(`shell脚本执行超时 (${timeoutMs / 1000}s)`));
            else reject(new Error(`shell脚本异常退出，exit code: ${code}`));
        });
        child.on("error", (err) => {
            if (err.killed) {
                reject(new Error(`shell脚本执行超时 (${timeoutMs / 1000}s)`));
            } else {
                reject(err);
            }
        });
    });
}

// ============================================================
// 主流程
// ============================================================

async function run() {
    try {
        // 读取输入参数
        const pr_id = core.getInput("pr_id", { required: true });
        const target_branch = core.getInput("target_branch", { required: true });
        const remote_url = core.getInput("remote_url", { required: true });
        const result_msg_path = core.getInput("result_msg_path");
        const step_id = core.getInput("step_id");

        // 获取脚本路径
        const scriptPath = path.resolve(__dirname, "./only_doc_check.sh");
        await fs.access(scriptPath);

        info("==============================================");
        info("  Only_Doc_Commit_Check Action");
        info("==============================================");
        info(`脚本路径:       ${scriptPath}`);
        info(`PR ID:          ${pr_id}`);
        info(`目标分支:       ${target_branch}`);
        info(`远程仓库URL:    ${remote_url}`);
        info(`结果路径:       ${result_msg_path}`);
        info(`步骤ID:         ${step_id}`);
        info("==============================================");

        // 执行 shell 脚本 (git clone + merge 可能较慢, 10分钟超时)
        const timeoutMs = 10 * 60 * 1000; // 10 minutes
        await execShellScript(
            scriptPath,
            [pr_id, target_branch, result_msg_path, step_id, remote_url],
            process.cwd(),
            timeoutMs
        );

        // --- 读取结果并设置输出 ---
        const resultDir = result_msg_path || ".";
        const resultFile = step_id
            ? path.join(resultDir, `${step_id}_result.json`)
            : path.join(resultDir, "only_doc_result.json");

        info("");
        info(`读取结果文件: ${resultFile}`);

        let isOnlyDoc = "unknown";
        try {
            const raw = fsSync.readFileSync(resultFile, "utf-8");
            const result = JSON.parse(raw);
            isOnlyDoc = result.is_only_doc || "unknown";
            info(`JSON 解析成功: is_only_doc = ${isOnlyDoc}`);
        } catch (readErr) {
            error(`读取结果文件失败: ${readErr.message}`);
        }

        // 使用工作流命令格式输出结果，供下游步骤引用
        setOutput("is_only_doc", isOnlyDoc);

        info("");
        info("Only_Doc_Commit_Check action completed successfully.");
    } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        setFailed(errorMessage);
    }
}

module.exports = { info, error, setFailed, setOutput, execShellScript, run };

// 仅在直接执行时运行 (非 require/import 时)
if (require.main === module) {
    run();
}
