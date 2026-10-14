# 在另一台 Windows 电脑测试 3C 路线

开发测试推荐通过 Git 同步同一个分支和提交。源码、声骸模型和模板资源随仓库一起获取，每台电脑分别建立 Python 环境。当前测试分支是 `feat/3c-echo-pathing`，包含局部地面路线首版；能力和限制见[声骸刷取说明](echo-farming.md)。

## 首次安装

准备 Windows 64 位电脑、Git、Python 3.12 和鸣潮 PC 客户端。首次游戏测试建议使用 1920×1080、16:9，并保持默认游戏亮度、关闭自动奔跑、使用北向固定的小地图。队伍中的每个角色都需要装备主声骸。

在 PowerShell 中进入想保存项目的目录，再执行：

```powershell
git clone --depth 1 --branch feat/3c-echo-pathing --recurse-submodules https://github.com/vv910/better-ww.git Better_ww
Set-Location Better_ww
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe main_debug.py
```

如果没有 `py` 命令，先用 `python --version` 确认是 Python 3.12，再用 `python -m venv .venv` 创建环境。之后仍使用 `.venv\Scripts\python.exe`。

程序应出现“🗺️ 按路线刷取3C声骸”和“📍 从地图录制3C路线”两个任务。如果没有，检查当前分支和提交：

```powershell
git branch --show-current
git rev-parse --short HEAD
```

`assets/echo_model/echo.onnx` 已纳入 Git。`ok_templates` 是子模块，漏拉子模块会导致相关模板/测试资源缺失，可执行：

```powershell
git submodule update --init --recursive
```

普通 GitHub 源码 ZIP 不包含子模块内容，建议按上述命令克隆。上游发布的安装包包含上游版本，测试本 fork 的 3C 功能时应运行这个分支的源码。

## 可选：先做离线检查

下面的截图测试使用仓库图片和不执行输入的测试后端，不需要打开游戏：

```powershell
.\.venv\Scripts\python.exe -m pip install pytest polib
.\.venv\Scripts\python.exe -u run_test.py tests.TestEchoRoute tests.TestFarm3CEchoTask tests.TestEchoFarmingRecovery
.\.venv\Scripts\python.exe -u run_test.py tests.Test3CRouteImages
.\.venv\Scripts\python.exe -u run_test.py tests.TestEcho
```

截图测试应分别启动进程；不要把多个继承 `TaskTestCase` 的模块合并到一次 `run_test.py` 调用中。

## 第一次游戏测试

选择一条短、平坦、能覆盖少量目标 3C 怪物的路线。大地图上用菱形标起点，用至少两个星标标出通行路径，运行“从地图录制3C路线”。检查生成 JSON 的点序和 `walk`/`combat`/`pickup` 动作，再让角色站到菱形附近，关闭地图，启动“按路线刷取3C声骸”。

首次重点观察三个结果：能否到达怪物点、能否完成战斗、能否吸收声骸并继续。当前还没有已标定的成品 3C 路线，首次运行需要创建路线。

如果要从另一台电脑复制已有路线，需要同时带上 JSON 和它引用的 `.map.png`，保持图片相对于 JSON 的位置。只复制 JSON 无法进行地图定位。记录器默认生成 `routes/3c/local-route.json` 和 `routes/3c/local-route.map.png`。

`.venv`、缓存、日志和个人设置不随 Git 同步。在新电脑重新建立环境并设置游戏热键。REA 是可选的源码研究工具，运行 3C 任务不需要安装 REA。

## 后续更新

在项目目录执行：

```powershell
git switch feat/3c-echo-pathing
git pull --ff-only
git submodule update --init --recursive
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main_debug.py
```

如果 `git pull` 提示本地文件有修改，先保留或提交自己的修改，再更新。`--ff-only` 会在分支发生分叉时停止，避免自动产生合并。

反馈测试问题时，附上 `git rev-parse --short HEAD`、游戏分辨率、队伍角色、路线文件、停在哪一步，以及 `logs/ok-ww.log` 和 `logs/ok-ww_error.log` 中对应时间的内容，便于复现同一个版本的问题。
