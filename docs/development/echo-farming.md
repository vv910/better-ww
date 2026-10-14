# 声骸刷取：4C 检查与 3C 路线扩展

## 4C 代码检查结论

检查基于 fork 的 `f9d808e6e3c60cb65c6e0535fc9fa9eac7e779d3`，以及本次修复。

`FarmEchoTask` 已具备传送/进本、自动战斗、战后拾取、重复挑战、死亡回血重进和月卡弹窗处理。默认使用 OpenVINO 的声骸模型位于 `assets/echo_model/echo.onnx`，模板、角色逻辑和截图测试资源也都存在。核心链路可以作为 3C 战斗和拾取的基础。

本次修复了三个会影响刷取的基础问题：

- 4C 进入大世界战斗后的异常重传没有重试上限；异常发生在轮次计数之前时可无限重传。现在受恢复预算限制，失败轮次也不会消耗重复刷取次数。
- `walk_to_yolo_echo` 在识别异常或取消时可能留下移动键。现在统一在 `finally` 中释放，并通过“吸收/Absorb”提示拾取声骸；未知窗口标题也不会取消文本过滤，避免把附近任意 F 交互算作声骸。
- `combat_once(wait_combat_time=0)` 的返回值未初始化；未找到敌人时仍会执行战斗收尾并增加战斗计数。现在先确认战斗入口。

这里确认的是源码和离线测试可用性。没有运行游戏中的 4C 验收，不能由此保证当前游戏版本的所有 Boss、分辨率和队伍组合都能稳定循环。Boss 图鉴数量、排序、界面坐标和部分专属逻辑仍依赖游戏版本。

## 3C 首版能力

任务列表新增“按路线刷取 3C 声骸”和“从地图记录 3C 路线”。默认路线执行一次。

执行链路为：校验路线和起点 → 小地图定位 → 按文件中的点序移动 → 遇敌暂停并战斗 → 搜索并吸收多个声骸 → 重新定位并继续。途经点、战斗点和专门拾取点可分别设置。

定位使用局部地图截图的 SIFT 特征及几何一致性验证，支持小地图与地图截图的缩放差异。定位失败时停下，连续 5 秒无法定位则退出；朝向不可信时退出；无进展时最多尝试两次脱困。每个路径点有超时，战斗和拾取也有时间预算。取消、异常或死亡时释放方向键和鼠标按键；死亡后停止，重新从起点启动。

“3C”由路线中的怪物点位决定。现有检测模型只识别掉落声骸，并不识别怪物种类或声骸 COST；途中遭遇的其他敌人也会被处理。

## 准备并运行路线

首次在另一台电脑运行时，请先按[跨电脑测试指南](testing-on-another-pc.md)克隆测试分支、初始化子模块并安装 Python 环境。

在仓库根目录启动源码版本：

```powershell
.\.venv\Scripts\python.exe main_debug.py
```

1. 使用鸣潮北向固定的小地图。当前仅针对地面路线；队伍需满足现有自动战斗的角色识别和主声骸配置要求。
2. 打开大地图，将待刷区域保持在同一张截图可见范围内。用菱形标记起点，用至少两个星标标出可通行路径；在障碍物、弯道附近增加途经点。菱形和星标应位于画面中央地图区域。
3. 运行“从地图记录 3C 路线”。默认保存 `routes/3c/local-route.json` 和 `local-route.map.png`；已有文件不会覆盖，换一个文件名即可保存新路线。
4. 打开 JSON 检查点序。记录器从菱形开始按最近邻生成初始顺序，这不保证地形上可通行。按实际道路调整 `waypoints` 的顺序，把普通途经点设为 `walk`，怪物点设为 `combat`，拾取点设为 `pickup`。
5. 角色站到菱形附近，关闭地图，在“按路线刷取 3C 声骸”的 `Route File` 中选择对应 JSON，然后启动任务。起点或地图匹配失败会停止，避免按错误位置行走。

路线使用截图像素坐标；图片和 JSON 配套。不能直接导入 BetterGI 的原神地图坐标，也不能用另一张缩放或裁剪不同的地图图片替换原图。

## 路线格式

下面仅说明字段，不是已标定的刷怪路线。坐标需要来自你自己的地图截图。

```json
{
  "schema_version": 1,
  "echo_cost": 3,
  "name": "我的局部 3C 路线",
  "map_image": "local-route.map.png",
  "map_region": [538, 130, 1670, 972],
  "start": {"x": 1200, "y": 470, "tolerance": 30},
  "waypoints": [
    {"x": 1140, "y": 540, "action": "walk", "tolerance": 12, "timeout": 120},
    {"x": 1090, "y": 590, "action": "combat", "tolerance": 10, "timeout": 120},
    {"x": 1030, "y": 620, "action": "pickup"}
  ]
}
```

`map_image` 相对于 JSON 所在目录解析，支持中文路径。`map_region` 是图片中的可见地图矩形 `[左, 上, 右, 下]`，可选，省略时使用全图。所有点必须在这个区域内。`tolerance` 为截图像素单位，默认 12；`timeout` 为该点移动时间预算，默认 120 秒，独立的战斗时间不计入移动预算，同一点最多处理中途遇敌五次。

## 当前边界与后续扩展

- 已实现局部地图、有序点位、地面寻路、自动战斗、多声骸搜索、有限脱困和异常退出。路线需要人工标定通行路径，程序不会自动规划绕开山体或建筑。
- 暂未实现完整世界地图拼接、跨区域自动传送、地图分层、攀爬/飞行/游泳动作、逐帧路线录制、死亡后的路线续跑，以及真实 3C 怪物/COST 分类。
- 声骸文本过滤支持简体中文、繁体中文和英文游戏界面；其他游戏语言需增加对应的吸收文字与验证素材。
- 战斗结束和声骸吸收确认复用原仓库的视觉判定；统计代表识别到的拾取操作，不是背包数量核对。
- 时间预算在视觉识别/角色动作的边界检查，正在执行的一次识别或技能可能使退出略晚于预算。

## 离线验证

开发环境使用仓库 `.venv`、Python 3.12.15，以及锁定的 `requirements.txt`。

本次 57 项路线/恢复/4C 逻辑测试和 17 项截图测试均通过，共 74 项；截图测试覆盖声骸模型、吸收 OCR、Boss 地图、战斗识别、小地图、朝向和真实星标导出。未进行游戏中的运行验收。

运行路线、恢复和 4C 逻辑测试：

```powershell
.\.venv\Scripts\python.exe -u run_test.py tests.TestEchoRoute tests.TestFarm3CEchoTask tests.TestEchoFarmingRecovery tests.TestFarmEchoPickup tests.TestFarmEchoTeleport tests.TestFarmEchoTaskEchoRate tests.TestDeathReentry tests.TestBaseCombatTask tests.TestDomainRecoveryLoop
```

截图测试需按仓库惯例逐个模块启动进程，避免 `TaskTestCase` 的全局截图捕获器在前一个模块销毁后失效：

```powershell
.\.venv\Scripts\python.exe -u run_test.py tests.TestFarmEcho
.\.venv\Scripts\python.exe -u run_test.py tests.TestEcho
.\.venv\Scripts\python.exe -u run_test.py tests.TestCombatCheck
.\.venv\Scripts\python.exe -u run_test.py tests.TestMap
.\.venv\Scripts\python.exe -u run_test.py tests.TestOCR
.\.venv\Scripts\python.exe -u run_test.py tests.Test3CRouteImages
```

路线测试覆盖顺序、动作分派、中途战斗、起点校验、定位丢失、缩放匹配、位置跳变拒绝、重复拾取、超时、取消、死亡和路线导出。定位测试包含仓库真实的鸣潮地图/小地图截图，另有合成地形缩放与异常输入测试。

## 参考与 REA

借鉴 [BetterGI 路径执行器](https://github.com/babalae/better-genshin-impact/tree/704d920d6d8fe09ae86628dab1e48464f0e4595e/BetterGenshinImpact/GameTask/AutoPathing) 的有序路径点、视觉闭环、有限脱困和动作分派思路；没有复制原神路线或素材。[脚本仓库](https://github.com/babalae/bettergi-scripts-list/tree/81bc0f00b41add3beb3e9ede9f88aac1714d0e5a) 只作为格式参考。

[REA](https://github.com/morluto/rea/tree/500e32399a3bc827e17fb4adaa97db3bc0df29b5) 是可选的源码研究工具，运行 3C 任务不依赖它。开发机已配置 `rea-agents` 6.4.0 MCP 和配套 skill；这些电脑级配置不会随 Git 同步。本次检查使用源码和本地离线测试，未通过 REA 执行游戏目标操作，也未安装 Hopper/Ghidra/IDA 等分析引擎。
