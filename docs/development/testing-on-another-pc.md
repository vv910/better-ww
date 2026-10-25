# 在另一台 Windows 电脑测试 3C 路线

只需要运行软件时，可直接下载[完整便携发布包](releases.md)，无需本页的 Python/Git 环境安装。以下流程用于开发和源码测试。

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
.\.venv\Scripts\python.exe -u run_test.py tests.TestRouteMarkers tests.TestRouteMarkerSelection tests.TestRouteSteering tests.TestMinimapHeading tests.TestMinimapTracker tests.TestMapFeatureRegions tests.TestEchoRoute tests.TestFarm3CEchoTask tests.TestEchoFarmingRecovery
.\.venv\Scripts\python.exe -u run_test.py tests.Test3CRouteImages
.\.venv\Scripts\python.exe -u run_test.py tests.TestEcho
```

截图测试应分别启动进程；不要把多个继承 `TaskTestCase` 的模块合并到一次 `run_test.py` 调用中。

## 第一次游戏测试

完整软件包请解压到英文路径，例如 `E:\BetterWW`；ok-script 会拒绝含中文的安装目录或父目录。源码调试目录也有同样限制。

选择一条短、平坦、能覆盖少量目标 3C 怪物的路线。在大地图上用星标标出通行路径，将目标星标组放在中央附近，并让附近已解锁信标可见。默认 `Star Selection = Connected Group` 按连接距离筛选远路线；若要录入中央区域内分散的整组星标，可显式选择 `All Visible Stars`，但必须先将其他路线移出中央录制区域。默认“最近信标 / Nearest Beacon”无需菱形，至少一个星标即可录制。运行“从地图录制3C路线”，默认开启 `Capture Map Context`，会在概览采集后打开菜单读取当前选中路径，再收起菜单，不会点击其他地区。需要时将 `Map Selection Path` 按父级到子级逐行填写精确菜单名称；手动路径只在录制时填一次，填写后跳过自动读取。保持“录制传送概览”开启，等待概览采集结束。检查 JSON 的点序和 `walk`/`combat`/`pickup` 动作，启动“按路线刷取3C声骸”。

首次重点观察三个结果：能否到达怪物点、能否完成战斗、能否吸收声骸并继续。当前还没有已标定的成品 3C 路线，首次运行需要创建路线。

录制后先打开同名 `.preview.png` 检查纳入的星标和点序。远处星标会受“相邻星标最大距离”和“录制范围半径”限制；信标到首个星标只受录制半径限制，需自己确认路径可通行。识别器以星形为目标，仍可能把地形或其他图标误认成星标，务必核对预览；误检时保留未叠加原始 `.map.png`、JSON 和日志。旧 `Diamond` 模式仍可手动放菱形起点，要求至少两个星标；没有传送信息的旧路线仍需手动站到起点。

启动后会先准备模板和声骸模型，准备期间请等待。当前默认持续移动，导航间隔 0.2 秒；先使用默认值。若路线里的拐弯只是经过位置，将相应点的 `action` 改成 `walk`，避免每个点都停下来搜怪和拾取。

执行带地图选择路径的路线时，任务先回到大世界，再由 OCR 唯一/稳定匹配地图菜单项并逐级点击；之后仍须通过强地图配准、信标/传送按钮确认和落地后小地图定位。梦枢天罗和黯原菜单已有截图回归，但不支持菜单滚动、任意地图身份推断或 `map_layer` 自动切层，也不等于已验收完整实机传送。自动搜索仍限于已录制概览、同一独立地图与同一地图层的视野范围。详细边界见[多条路线与自动传送](echo-farming.md#route-queue)。

如果要从另一台电脑复制已有路线，需要同时带上 JSON、`.map.png`、`.preview.png` 和全部生成的 `.view-N.png`，保持图片相对于 JSON 的位置。只复制 JSON 无法进行地图定位。默认路线文件名为 `routes/3c/local-route.json`；已有文件不会覆盖，请用新名称录制。

需要离地开翼保护时，在开始前手动选择控物。默认不会打开工具轮盘；未选择控物时会提示本条路线未启用防坠保护并继续。可选开启自动选择控物，准备失败时同样降级继续，不能保证所有坠落都能防止。

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

如果提示 `Minimap position lost for 5 seconds` 或 `Cannot confirm route start`，表示当前位置没有通过地图匹配或起点距离检查。2026-10-24 的改进会在异常退出时把当前路线快照、引用地图副本及失败前连续小地图保存到 `logs/3c-replays/`。日志出现“已保存寻路复现包”后，直接点击开发工具的“导出日志”，ZIP 会包含该包；导出功能本身不会遍历整个 `routes` 目录。若日志提示包未保存或资产不完整，再手动加入该路线 JSON、`.map.png` 和全部 `.view-N.png` 原图。

如果提示 `Cannot open map for route entry`，表示地图界面的视觉确认超时，还没有进行路线定位。2026-10-25 起同时支持旧黄色定位图钉和右上回中、缩放加减按钮的新布局；入口及中途地图恢复失败会明确保存当次检查画面，并在复现包中生成独立 `map_ui` 事件。请一并提供 `screenshots` 中对应的失败图片。仅凭错误文字不能断言地图键没生效；即使没有小地图事件，也可能只是任务在采样前已停止。

如果提示 `Cannot find route map bookmark`，请保留 `3c-teleport-bookmark-mismatch` 等搜索截图。视图闭环改造会在日志中记录 `3C map search` 的阶段、参考匹配、缩放/拖动实际变化，以及弹窗恢复结果，并写入 `bookmark:` 开头的 `map_ui` 事件。请先导出或上传同一次运行的日志；通常不需仅为诊断重新录制路线。若失败图已是“自定义标记”面板，应先分析界面恢复和输入，不应放宽地形匹配阈值。

静态诊断也继续保留：`last-attempt-minimap` 是最后一次小地图匹配实际处理的输入，日志中的 `last_minimap_diagnostic` 保留其匹配结果，不会被随后的大地图恢复覆盖；`minimap` 是报错时画面的小地图，`last-good` 是最后一次成功定位的小地图；`start-` 表示在起点检查阶段失败。先导出同一次运行的日志再重启，无需仅为了诊断重新录制路线。复现包及离线回放命令见[导航调研与改进](pathing-research.md#navigation-replay)。

排查没有自动传送时，先检查任务的“路线文件”和“路线队列”：队列非空会覆盖单路线文件，多个文件也可能使用相同显示名称。日志 `3C route selection` 记录实际文件绝对路径、设置来源和正在执行的源码路径，`3C route entry plan` 记录实际起点、图片、传送视图数及地图菜单路径。首条旧式路线可以没有 `teleport`，此时会明确提示从当前位置开始，不会自动传送；有 `teleport.views` 时 `anchor` 可以为空。若日志文件名或起点与提供的 JSON 不同，应先核对实际选中的文件，再分析截图失配。更新源码后须关闭旧进程、重新启动才能生效。

`3C Stage` 可区分移动、战斗和搜索声骸。所有标记都录成 `combat` 时，每一点都会停下检查战斗并搜索声骸；纯转弯位置应改为 `walk`。短暂 SIFT 失配由最多 3 秒、60 个地图像素的光流衔接；独立全图地形复核通过后可建立新绝对定位，复核失败不会延长光流预算。近点会减速，最终到点需要停步后的新画面绝对确认；不能仅凭一次恢复运行就判断配套地图的持续匹配已解决。
