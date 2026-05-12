# Codex RPC Bridge 使用说明

这个目录提供一个轻量 HTTP RPC 服务，用于让外部 AI agent 操作腾讯开悟 / code-server 开发容器中的代码。

适用场景：

- 容器不能访问公网。
- 容器不能正常上传本地文件。
- 外部 AI 无法直接 SSH 进入容器。
- 但浏览器里的 Tencent Arena IDE 已经连接到容器，并且可以通过 code-server 的端口代理访问容器端口。

目录名使用 `codex_rpc_bridge`，不是点开头目录，方便手动复制。运行时 token、日志和备份不要提交到仓库。

## 文件说明

- `codex_file_rpc.py`：RPC 服务脚本。
- `AI_USAGE.md`：本说明文档。
- 运行时会在目标环境中生成：
  - `token`
  - `admin_token`
  - `codex_file_rpc.log`
  - `backups/`

建议把运行时文件放在：

```text
codex_rpc_bridge_runtime/
```

不要把 token、admin_token、日志、备份提交到仓库。

## 在腾讯容器中启动

在容器工作区根目录运行：

```bash
cd /data/projects/legged_robot_competition_26 || exit 1
mkdir -p codex_rpc_bridge_runtime
```

生成普通 token 和 admin token：

```bash
python3 - <<'PY'
from pathlib import Path
import secrets

for name in ("token", "admin_token"):
    p = Path("codex_rpc_bridge_runtime") / name
    if not p.exists() or not p.read_text().strip():
        p.write_text(secrets.token_urlsafe(32) + "\n")
    print(name + "=" + p.read_text().strip())
PY
```

启动服务：

```bash
export CODEX_RPC_TOKEN="$(cat codex_rpc_bridge_runtime/token)"
export CODEX_RPC_ADMIN_TOKEN="$(cat codex_rpc_bridge_runtime/admin_token)"

python3 - <<'PY'
import os, signal
from pathlib import Path

for p in Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        continue
    if "codex_file_rpc.py" in cmd:
        try:
            os.kill(int(p.name), signal.SIGTERM)
        except ProcessLookupError:
            pass
PY

sleep 1

nohup python3 codex_rpc_bridge/codex_file_rpc.py \
  --host 0.0.0.0 \
  --port 8765 \
  --root /data/projects/legged_robot_competition_26 \
  --token "$CODEX_RPC_TOKEN" \
  --admin-token "$CODEX_RPC_ADMIN_TOKEN" \
  > codex_rpc_bridge_runtime/codex_file_rpc.log 2>&1 &

sleep 1
cat codex_rpc_bridge_runtime/codex_file_rpc.log
curl -s http://127.0.0.1:8765/api/health
echo
```

如果成功，会看到类似：

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26",
  "auth": true,
  "admin_auth": true
}
```

## 外部访问地址

在腾讯开悟 code-server 代理下，外部访问地址通常是：

```text
https://tencentarena.com/p5/ide/<redacted>/proxy/<redacted>
```

其中 `11428` 可能会随 IDE 会话变化。如果访问不通，应查看当前 IDE iframe 地址，使用其中的 `/p5/ide/<id>/proxy/<redacted>`。

### 如何判断 BASE 地址

`BASE` 指的是 code-server 把容器内端口代理到浏览器可访问地址后的前缀：

```text
https://tencentarena.com/p5/ide/<IDE_ID>/proxy/<redacted>
```

容器内 RPC 服务只知道自己监听：

```text
http://0.0.0.0:8765
```

它通常无法自动知道外部浏览器侧的 `BASE`，因为 `BASE` 是腾讯开悟 / code-server 在页面层提供的代理路径。

判断方式：

1. 打开腾讯开悟 IDE 页面。
2. 找到 code-server iframe 的真实地址，通常形如：

```text
https://tencentarena.com/p5/ide/<redacted>/?folder=/data/projects/legged_robot_competition_26
```

3. 取其中的：

```text
https://tencentarena.com/p5/ide/<redacted>
```

4. 拼接：

```text
/proxy/<redacted>
```

得到：

```text
https://tencentarena.com/p5/ide/<redacted>/proxy/<redacted>
```

5. 用健康检查验证：

```bash
BASE='https://tencentarena.com/p5/ide/<redacted>/proxy/<redacted>'
curl -s "$BASE/api/health"
```

如果返回类似下面内容，说明 `BASE` 正确：

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26"
}
```

如果 `/api/health` 访问不到，优先检查：

- IDE 页面是否仍然登录。
- 容器内 RPC 服务是否仍在运行。
- `IDE_ID` 是否变化。
- 端口是否仍是 `8765`。

也可以手动指定 `IDE_ID` 并打印候选地址：

```bash
IDE_ID=11428
PORT=8765
echo "BASE=https://tencentarena.com/p5/ide/${IDE_ID}/proxy/${PORT}"
```

### 浏览器窗口与端口转发信息

外部 AI 能否自动读取端口转发信息，取决于它是否能控制或连接到当前浏览器。

Codex 当前能读取由 `agent-browser` 创建的浏览器窗口，是因为该窗口通过 Chrome DevTools Protocol 暴露了页面状态。AI 可以读取当前 URL、页面文本、按钮、code-server 页面和端口转发面板。

普通浏览器默认不能被 Codex 读取。要让普通 Chrome 也能被 Codex 读取，需要用远程调试端口启动 Chrome，或者始终让 Codex 使用同一个持久化浏览器 profile 打开页面。

推荐方式一：让 Codex 创建并复用同一个 Tencent Arena 窗口。

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

含义：

- `AGENT_BROWSER_SESSION=tencent-arena` 固定浏览器会话名。
- `AGENT_BROWSER_SESSION_NAME=tencent-arena` 保存并复用 cookies / localStorage。
- 如果登录状态仍有效，Codex 可以直接进入 IDE。
- 如果登录状态失效，需要手动登录一次，之后 Codex 可以继续读取页面状态。

登录完成后，也可以保存浏览器状态：

```bash
agent-browser state save ./tencent-arena-auth.json
```

后续用保存的状态打开：

```bash
agent-browser --state ./tencent-arena-auth.json open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

推荐方式二：让普通 Chrome 暴露远程调试端口。

macOS 示例：

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.chrome-codex-cdp" \
  --remote-allow-origins=* \
  "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

然后让 Codex 连接这个浏览器：

```bash
agent-browser connect 9222
```

或：

```bash
agent-browser --cdp 9222 snapshot -i
```

不建议直接使用日常 Chrome 的默认 profile。建议使用独立目录：

```bash
--user-data-dir="$HOME/.chrome-codex-cdp"
```

原因：

- 默认 Chrome profile 可能已经被普通 Chrome 占用。
- 远程调试会让页面可被自动化读取，独立 profile 更容易控制风险。
- 登录一次后，这个独立 profile 会保留登录状态。

如果只是为了得到 RPC 的 `BASE`，通常不需要读取端口转发面板。只要知道 IDE 页面中的 `/p5/ide/<IDE_ID>` 和容器端口，就可以直接拼接：

```text
https://tencentarena.com/p5/ide/<IDE_ID>/proxy/<PORT>
```

例如：

```text
https://tencentarena.com/p5/ide/<redacted>/proxy/<redacted>
```

然后用 `/api/health` 验证即可。

### 其他 Codex 如何实际操作

腾讯开悟的 `/proxy/<PORT>` 通常要求浏览器登录态。外部 Codex 直接用
`curl "$BASE/api/health"` 时，如果返回下面内容，说明请求被 Tencent Arena
代理层拦截，还没有到达容器里的 RPC 服务：

```json
{"code":1040,"msg":"token 校验失败","reason":"CommonErrors.TOKEN_NOT_VALID"}
```

这种情况下，RPC token 本身无法被校验。应让 Codex 先通过可自动化浏览器获得
Tencent Arena 登录态，再在页面内调用 RPC。

推荐流程如下。

1. 让 Codex 使用浏览器自动化 skill。

   如果当前 Codex 环境有 `agent-browser` skill，应先读取该 skill 的说明，再启动
   一个可复用浏览器会话：

   ```bash
   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser --headed open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
   ```

   如果 `--headed` 被提示已忽略，说明已有无头 daemon 在运行。可以先关闭再重开：

   ```bash
   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser close

   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser --headed open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
   ```

2. 用户在弹出的浏览器中登录腾讯开悟，并进入 IDE。

3. Codex 通过同一个浏览器会话打开 health 地址：

   ```bash
   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser open "$BASE/api/health"

   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser get text body
   ```

   成功时应看到：

   ```json
   {
     "ok": true,
     "version": "0.7.1",
     "root": "/data/projects/legged_robot_competition_26",
     "auth": true,
     "admin_auth": true
   }
   ```

4. 用页面内 `fetch` 调用需要 token 的 RPC 接口。

   由于请求由已登录浏览器发起，会携带 Tencent Arena 登录态；同时通过 header
   携带 RPC token：

   ```bash
   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser eval "fetch('$BASE/api/token-check', {
     headers: {'X-Codex-Token': '<normal token>'}
   }).then(r => r.text())"
   ```

   成功时返回：

   ```json
   {"ok": true, "role": "normal"}
   ```

5. 读取文件示例：

   ```bash
   AGENT_BROWSER_SESSION=tencent-arena \
   AGENT_BROWSER_SESSION_NAME=tencent-arena \
   agent-browser eval "fetch('$BASE/api/read/train_test.py', {
     headers: {'X-Codex-Token': '<normal token>'}
   }).then(r => r.text())"
   ```

6. 写文件时仍然必须遵守 `read -> expected_sha256 -> write_b64 -> read back`
   的流程。`fetch` 调用和 `curl` 调用的 URL 与参数完全一致，只是执行位置从
   命令行变成了已登录浏览器页面。

安全注意：

- 不要把真实 token 写进代码、日志、提交记录或截图。
- 某些 Codex 环境会阻止把 token 发送到 `tencentarena.com` 这样的外部域。
  即使该 URL 最终代理到容器，也应先让用户明确授权。
- 普通读写只使用 normal token；只有确实需要执行命令时才请求 admin token。
- 如果需要保存浏览器登录态，可使用：

  ```bash
  AGENT_BROWSER_SESSION=tencent-arena \
  AGENT_BROWSER_SESSION_NAME=tencent-arena \
  agent-browser state save ./tencent-arena-auth.json
  ```

## 认证方式

普通 token：

```text
X-Codex-Token: <codex_rpc_bridge_runtime/token 的内容>
```

Admin token：

```text
X-Codex-Admin-Token: <codex_rpc_bridge_runtime/admin_token 的内容>
```

## 权限模型

普通 token：

- 可以全局读取、列目录、查看 stat、tree。
- 可以在工作区内写入、追加、创建目录、touch、删除文件或空目录。
- 不能读取运行时 token 文件。
- 不能执行 shell 命令。
- 不能写出工作区。
- 不能通过 symlink 写出工作区。

Admin token：

- 可以全局读取。
- 可以全局写入、追加、删除。
- 可以执行 shell 命令。
- 可以停止 RPC 服务。

全局绝对路径写法：

```text
__abs__/etc/hostname
__abs__/tmp/file.txt
```

工作区路径写法：

```text
train_test.py
agent_diy/agent.py
```

## 稳定接口

```text
GET /api/health
GET /api/token-check
GET /api/list
GET /api/list/<dir>
GET /api/tree?depth=N
GET /api/tree/<dir>?depth=N
GET /api/stat/<path>
GET /api/read/<file>
GET /api/read/<file>?encoding=base64
GET /api/write_text/<file>?text=<urlencoded-text>
GET /api/write_b64/<file>?data=<base64url>
GET /api/append_b64/<file>?data=<base64url>
GET /api/mkdir_get/<dir>
GET /api/touch/<file>
GET /api/delete_get/<file-or-empty-dir>
GET /api/exec_b64?cwd=<dir>&timeout=<seconds>&cmd=<base64url-command>
POST /api/stop
```

说明：

- `exec_b64` 需要 admin token。
- `stop` 需要 admin token。
- 删除接口只删除文件或空目录，不递归删除非空目录。
- 推荐使用 `write_b64` 写源码文件，不推荐用 `write_text` 写大段代码。

## 外部 AI 推荐工作流

### 1. 读取文件

```bash
BASE='https://tencentarena.com/p5/ide/<redacted>/proxy/<redacted>'
TOKEN='<normal token>'

curl -s \
  -H "X-Codex-Token: $TOKEN" \
  "$BASE/api/read/train_test.py"
```

返回中会包含：

```json
{
  "ok": true,
  "path": "train_test.py",
  "size": 1274,
  "sha256": "...",
  "encoding": "utf-8",
  "text": "..."
}
```

### 2. 使用 expected_sha256 写回

写文件前应先读取旧文件，并保存返回的 `sha256`。写入时带上 `expected_sha256`，防止覆盖用户或其他 agent 的并发修改。

```bash
DATA_B64="$(python3 - <<'PY'
import base64
text = "new content\n"
print(base64.urlsafe_b64encode(text.encode()).decode().rstrip("="))
PY
)"

curl -s \
  -H "X-Codex-Token: $TOKEN" \
  "$BASE/api/write_b64/path/to/file.py?expected_sha256=$OLD_SHA&data=$DATA_B64"
```

如果文件在读取后被别人改过，会返回错误：

```json
{
  "ok": false,
  "error": "file changed since read: expected_sha256=... actual_sha256=..."
}
```

### 3. 写完后读回验证

```bash
curl -s \
  -H "X-Codex-Token: $TOKEN" \
  "$BASE/api/read/path/to/file.py"
```

### 4. 执行检查命令

执行 shell 命令需要 admin token。

```bash
ADMIN_TOKEN='<admin token>'

CMD_B64="$(python3 - <<'PY'
import base64
cmd = "python3 -m py_compile train_test.py"
print(base64.urlsafe_b64encode(cmd.encode()).decode().rstrip("="))
PY
)"

curl -s \
  -H "X-Codex-Admin-Token: $ADMIN_TOKEN" \
  "$BASE/api/exec_b64?cwd=.&timeout=30&cmd=$CMD_B64"
```

返回示例：

```json
{
  "ok": true,
  "cwd": ".",
  "cmd": "python3 -m py_compile train_test.py",
  "timeout": 30,
  "returncode": 0,
  "stdout": "",
  "stderr": "",
  "role": "admin"
}
```

## 可靠性机制

服务包含以下保护：

- `/api/read` 返回 `sha256`。
- 写入接口支持 `expected_sha256`，可避免覆盖并发修改。
- 覆盖、追加、删除已有文件前会自动备份。
- 备份路径：

```text
codex_rpc_bridge_runtime/backups/<timestamp>/<path>.bak
```

- 覆盖写使用临时文件 + 原子替换。
- 普通 token 不能通过 symlink 写出工作区。
- 删除接口不递归删除非空目录。

## 给 AI Agent 的要求

外部 AI agent 应遵守：

1. 先 `read`，再修改。
2. 写入时必须带 `expected_sha256`。
3. 写入后必须再次 `read` 验证。
4. 写源码优先使用 `write_b64`。
5. 不要并发写同一个文件。
6. 普通开发只用 normal token。
7. 只有需要执行命令或全局写入时才使用 admin token。
8. 不要把 token 写入日志、代码、提交记录或公开消息。

## 停止服务

```bash
curl -s \
  -H "X-Codex-Admin-Token: $ADMIN_TOKEN" \
  -X POST \
  "$BASE/api/stop"
```

## .gitignore

本仓库已经在 `.gitignore` 中忽略运行时目录。若复制到其他项目，也应加入：

```gitignore
codex_rpc_bridge_runtime/
```
