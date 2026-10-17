# 在另一台 Windows 电脑测试 3C 路线

开发测试推荐通过 Git 同步同一个分支和提交。源码、声骸模型和模板资源随仓库一起获取，每台电脑分别建立 Python 环境。当前测试分支是 `feat/3c-echo-pathing`，包含局部地面路线首版；能力和限制见[声骸刷取说明](echo-farming.md)。

## 已经在 WSL 中克隆了仓库

如果提示符类似 `vv@DESKTOP:/mnt/e/Better_ww$`，当前是 WSL/Linux 的 Bash。下面的 `py` 和 `.venv\Scripts\python.exe` 命令需要 Windows PowerShell；游戏自动化依赖 Windows API 和 Windows 版 Python，不能通过安装 WSL 的 `python3` 或终端提示的 `pythonpy` 来启动。

仓库和子模块已经下载成功时，无需重新克隆。Windows Terminal 中切换到“PowerShell”或“Windows PowerShell”，或者从 Windows 开始菜单打开 PowerShell。`/mnt/e/Better_ww` 对应 Windows 的 `E:\Better_ww`，在 Windows PowerShell 中执行：

```powershell
Set-Location 'E:\Better_ww'
py -3.12 --version
```

如果 Windows 中尚未安装 Python 3.12，可在这个 PowerShell 窗口安装：

```powershell
winget install --exact --id Python.Python.3.12 --source winget
```

安装后关闭并重新打开 Windows PowerShell，再执行 `py -3.12 --version`。没有 WinGet 时，可使用 [Python 3.12 的 Windows 64 位安装程序](https://www.python.org/downloads/release/python-31210/)；保留 Python Launcher 的安装选项。

确认版本为 3.12 后，在已有仓库目录执行：

```powershell
Set-Location 'E:\Better_ww'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe main_debug.py
```

这些步骤使用已经下载的源码和子模块，每台电脑建立自己的 Windows 虚拟环境。WinGet 用法见[微软安装说明](https://learn.microsoft.com/en-us/windows/package-manager/winget/install)，包标识可在[官方包清单](https://github.com/microsoft/winget-pkgs/tree/master/manifests/p/Python/Python/3/12)核对。

## 首次安装

准备 Windows 64 位电脑、Git、Python 3.12 和鸣潮 PC 客户端。首次游戏测试建议使用 1920×1080、16:9，并保持默认游戏亮度、关闭自动奔跑、使用北向固定的小地图。队伍中的每个角色都需要装备主声骸。

在 **Windows PowerShell** 中进入想保存项目的目录，再执行；不要粘贴到 WSL/Ubuntu/Bash 中：

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
.\.venv\Scripts\python.exe -u run_test.py tests.TestMinimapHeading tests.TestEchoRoute tests.TestFarm3CEchoTask tests.TestEchoFarmingRecovery
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

如果提示 `Cannot confirm minimap player heading after 3 frames`，日志会包含画面尺寸、匹配分数和方向分数差；请同时保留“截图目录”中名称包含 `3c-heading-minimap` 的图片。程序重启可能清理旧截图，先保存诊断图片再重启。现有路线可以继续使用，不需要因朝向识别失败重新录制。
