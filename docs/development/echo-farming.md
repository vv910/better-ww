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

任务列表新增“按路线刷取 3C 声骸”和“从地图记录 3C 路线”。支持单条路线，也支持多条路线按队列传送并依次执行，各执行一次。

执行链路为：校验路线和起点 → 小地图定位 → 按文件中的点序移动 → 遇敌暂停并战斗 → 搜索并吸收多个声骸 → 重新定位并继续。途经点、战斗点和专门拾取点可分别设置。

定位使用局部地图截图的 SIFT 特征及几何一致性验证，支持小地图与地图截图的缩放差异。首次和战后必须通过 SIFT；移动中用 LK 光流跟踪地形，并约每 0.5 秒重新尝试 SIFT 校正。光流最多衔接最后一次绝对定位后的 3 秒、60 个录制地图像素，要求双向一致、至少 12 个内点且分布覆盖小地图；不会用按键时长推算位置。匹配冲突、图像突变或超出预算时停止接管。朝向使用小地图中心的金色箭头轮廓，统一尺寸后搜索角度与大小；同时检查匹配分数和其他方向的分数差，避免把圆形图标当作箭头。两种位置识别都失败时松键，连续 5 秒无法定位则退出；朝向不可信时取新帧重试，连续 3 帧失败则保存小地图截图并退出；无进展时最多尝试两次脱困。每个路径点有超时，战斗和拾取也有时间预算。取消、异常或死亡时释放方向键和鼠标按键；死亡后停止，重新从起点启动。

参考特征先从 `map_region` 裁图提取，再把关键点偏移回截图坐标；传送的大地图投影也使用同样处理。SIFT 的特征数限额在遮罩过滤之前生效，仅传入整图遮罩会让区域外 UI 和高纹理内容挤占地图特征额度。先裁图保证 8000 个参考特征的预算只用于有效地图，不调整几何内点、比例、空间跨度、缩放、旋转和跳变门槛。有效区域外的界面变化不会改变参考特征，路线 JSON 坐标及配套图片格式保持兼容。

移动使用持续按键：默认每 0.2 秒更新位置和方向，只有方向变化才切换按键，保留相邻方向的共同按键。普通 `walk` 点之间保持移动；抵达 `combat`/`pickup` 点、定位丢失、朝向不可靠或遇敌时松键。相机在出发、战斗后或较长暂停恢复时校准一次；1 秒以内的定位暂停保留已校准的相机方向，恢复时不再重复等待约 1.2 秒。8 方向选择带迟滞，并在按键稳定后用角色箭头估算相机朝向，适应缓慢的自动回正；游戏中快速自动转镜、角色受击转身仍需实测。

任务信息中的 `3C Stage` 显示当前处于移动、战斗或搜索声骸。战后或空战斗点仍搜索四个方向，但每次相机转向只等待 0.4 秒再识别，避免空场扫描中的四次长等待；找到声骸后的接近和吸收仍沿用原有逻辑及剩余时间预算。搜索结束会清空旧定位，继续移动前重新匹配地图。

启动会先显示识别准备日志，预加载模板、识别队伍并编译/预热声骸模型，再开始移动。3C 执行期间 OpenCV 最多使用两个工作线程，任务退出后恢复原值；OpenVINO CPU 推理最多使用四个线程，减少与游戏争抢 CPU。`Navigation Interval`（导航刷新间隔）支持 0.1–0.5 秒，默认 0.2；降低间隔会增加 CPU 占用。模型首次编译仍需要时间，准备阶段请等待“3C识别准备完成”。

“3C”由路线中的怪物点位决定。现有检测模型只识别掉落声骸，并不识别怪物种类或声骸 COST；途中遭遇的其他敌人也会被处理。

## 准备并运行路线

首次在另一台电脑运行时，请先按[跨电脑测试指南](testing-on-another-pc.md)克隆测试分支、初始化子模块并安装 Python 环境。

在仓库根目录启动源码版本：

```powershell
.\.venv\Scripts\python.exe main_debug.py
```

1. 使用鸣潮北向固定的小地图。当前仅针对地面路线；队伍需满足现有自动战斗的角色识别和主声骸配置要求。
2. 打开大地图，将当前一条路线放在画面中央。用一个菱形标记起点，用至少两个星标标出可通行路径；在障碍物、弯道附近增加途经点。其他类型的图标不会加入路线。菱形上半部应可见；同一视野内有多个菱形时会停止并提示调整视野。
3. 运行“从地图记录 3C 路线”。保持当前缩放，不自动点击缩放滑条。默认保存 `routes/3c/local-route.json`、识别用原图 `local-route.map.png` 和检查用 `local-route.preview.png`；三者都不会覆盖已有文件，换一个文件名即可保存新路线。
4. 先检查预览，再检查 JSON 点序。预览中 `S` 是起点，绿色数字/连线是纳入路线的星标顺序，红叉是未纳入的星标。记录器从菱形开始按最近邻生成初始顺序，这不保证地形上可通行。按实际道路调整 `waypoints` 的顺序，把普通途经点设为 `walk`，怪物点设为 `combat`，拾取点设为 `pickup`。
5. 角色站到菱形附近，关闭地图，在“按路线刷取 3C 声骸”的 `Route File` 中选择对应 JSON，然后启动任务。起点或地图匹配失败会停止，避免按错误位置行走。

路线使用截图像素坐标；图片和 JSON 配套。不能直接导入 BetterGI 的原神地图坐标，也不能用另一张缩放或裁剪不同的地图图片替换原图。

录制会限制点位范围，避免把其他路线的远处星标串进来：

- “相邻星标最大距离 / Maximum Marker Gap”默认 200。从当前点选择距离不超过此值的最近星标，找不到下一点即结束。
- “录制范围半径 / Recording Radius”默认 500。只考虑起点周围这个半径内的星标，防止沿一串星标不断延伸到另一条路线。
- 两个数值均以 1080p 像素为基准，按截图高度换算；2560×1440 下默认约为 267 和 667 个截图像素。它们不是游戏米数，地图缩放变化后对应的地理范围也会变化。

如果近邻的两条路线仍被连接，调小相邻距离或录制半径，并检查预览。任务日志会记录识别到的菱形/星标数、匹配分数以及纳入/排除数量。完全找不到起点时保留大地图并保存 `3c-record-marker-failed` 截图；星标不足两个时保存 `3c-record-markers` 诊断预览。不会把骷髅、叶子、任务图标等自动转换成星标。

## 多条路线与自动传送 {#route-queue}

在“路线队列 / Route Queue”中每行填写一个路线 JSON，按行序执行。留空时仍使用原来的“Route File”，旧文件不需要为了连续移动重新录制。例如：

```text
routes/3c/route-a.json
routes/3c/route-b.json
routes/3c/route-c.json
```

分散路线需要各自的地图截图和传送信息。录制时在“传送定位Boss / Teleport Anchor”中填写附近**固定位置的 Boss 图鉴名称**，使用游戏当前语言。起点菱形应靠近已解锁信标（50 个录制地图像素以内），避免覆盖信标图标。记录器会保存最近信标在截图中的坐标；找不到近邻信标则拒绝保存。

选 Boss 的依据是：通过它的图鉴“探测”打开地图后，路线入口信标和录制区域应在同一可见地图区域内。大地图视图匹配支持一定的缩放和平移，尚不支持自动平移到视口外的信标、跨地图层切换，或跨区域寻找远离这个 Boss 的信标。优先测试短路线，再增加区域。

执行每条带传送信息的路线时，任务先通过 F2 全部怪物图鉴搜索 Boss、探测打开地图，将当前大地图与该路线截图做特征匹配，再确认投影位置附近存在信标图标才点击传送。回到大世界后清空上一位置并验证路线起点。地图匹配、信标选择或起点验证失败时停止队列；错误图像以 `3c-teleport-` 开头保存。

队列的第二条及后续路线必须包含 `teleport`，程序会在输入游戏前检查。第一条可以没有，此时需要手动站到起点；带有 `teleport` 的第一条也会自动传送。队列不自动续跑或重复，取消/失败后重新启动会从第一条开始。

旧路线也可以补充下面的 `teleport` 字段：`x/y` 必须是**配套地图截图**中的信标坐标，并在 `start.tolerance` 范围内；不能使用另一张地图的坐标。无法确认坐标时，建议以新文件名录制一条带传送信息的路线。

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
  "teleport": {"anchor": "无冠者", "x": 1206, "y": 466},
  "waypoints": [
    {"x": 1140, "y": 540, "action": "walk", "tolerance": 12, "timeout": 120},
    {"x": 1090, "y": 590, "action": "combat", "tolerance": 10, "timeout": 120},
    {"x": 1030, "y": 620, "action": "pickup"}
  ]
}
```

`map_image` 相对于 JSON 所在目录解析，支持中文路径。`map_region` 是图片中的可见地图矩形 `[左, 上, 右, 下]`，可选，省略时使用全图。所有点必须在这个区域内。`tolerance` 为截图像素单位，默认 12；`timeout` 为该点移动时间预算，默认 120 秒，独立的战斗时间不计入移动预算，同一点最多处理中途遇敌五次。

## 当前边界与后续扩展

- 已实现局部地图、有序点位、持续按键移动、路线队列、Boss 图鉴定位后的信标传送、自动战斗、多声骸搜索、有限脱困和异常退出。路线需要人工标定通行路径，程序不会自动规划绕开山体或建筑。
- 暂未实现完整世界地图拼接、任意世界坐标传送、视口外地图平移搜索、地图分层、攀爬/飞行/游泳动作、逐帧路线录制、死亡后的路线续跑，以及真实 3C 怪物/COST 分类。
- 声骸文本过滤支持简体中文、繁体中文和英文游戏界面；其他游戏语言需增加对应的吸收文字与验证素材。
- 战斗结束和声骸吸收确认复用原仓库的视觉判定；统计代表识别到的拾取操作，不是背包数量核对。
- 时间预算在视觉识别/角色动作的边界检查，正在执行的一次识别或技能可能使退出略晚于预算。

## 离线验证

开发环境使用仓库 `.venv`、Python 3.12.15，以及锁定的 `requirements.txt`。

基础版本的 57 项路线/恢复/4C 逻辑测试和 17 项截图测试均通过，共 74 项。2026-10-17 测试电脑首次运行时，起点定位得到 49 个有效特征，随后因旧 RGB 箭头匹配未通过置信度校验而停止。已改用箭头轮廓匹配和有限重试，完整游戏路线仍待复测。

本次朝向修复另通过 21 项逻辑测试和 4 项截图测试，包含暂时失败后的重试、持续失败停止、战斗打断、取消释放按键，以及真实小地图截图在 1080p、缩放至 1440p 和 2160p 下的识别。缩放截图测试不等于这些分辨率下的实机运行验收。

随后测试日志确认可以移动，但每轮约 1.9 秒中只走 0.3 秒。持续移动与传送队列改动共通过 65 项针对性离线测试，覆盖按键保持、8 方向/相机漂移模拟、队列顺序、传送失败、录制信标、真实地图缩放平移、被动遇敌检测以及实际声骸模型；游戏中的连续移动和跨区域传送仍需复测。开发机用 1080p 真实截图测试，限制两个 OpenCV 线程后单次位置与方向计算均值约 73 ms；这不包含截图采集与战斗检查，也不代表测试电脑的实际帧率。

录制器另加入玩家箭头/信标遮挡起点的真实回归图（已裁去账号UI）。旧 RGB 匹配约 0.63，未达到 0.7 门槛；白色轮廓与菱形上半部联合识别可定位同一个起点，并在 1080p、1440p、2160p 缩放下保持三个星标。完整录制测试加入两个远处星标，确认它们被排除并保存点序预览；普通地图与已有路线导出也继续通过。本次录制器修复共通过 38 项逻辑测试与 8 项截图测试，合计 46 项。

定位失败诊断会记录小地图特征数、有效匹配数、几何内点、缩放、旋转和位置跳变。失败原因区分 `too_few_minimap_features`、`too_few_matches`、`minimap_not_north_up`、`scale_out_of_range`、`position_jump` 和 `position_outside_recorded_map` 等；局部与全图两次搜索各保留一次诊断。起点检查或连续 5 秒失配时自动保存当前画面、小地图和最后一次成功匹配的小地图，文件以 `3c-location-start-` 或 `3c-location-lost-` 开头。这些诊断不改变原来的匹配门槛和停止规则，不能仅凭“位置丢失”日志判断具体原因。

连续移动修复通过 67 项定位/导航/任务逻辑测试和 10 项截图测试。覆盖相对跟踪的累计时间/距离限制、冲突时停止、战后重定位、短暂停顿不重置视角，以及真实小地图素材在三种尺寸下的平移跟踪。

收到配套地图后，复现了有效区域特征被整图限额挤占的问题：原始图的有效区域在整图遮罩方式下只保留 2687 个特征；先裁有效区域可使用完整的 8000 个预算，同一张失败小地图由 4 个几何内点提高到 26 个，内点比例约 0.81。原图中账号 UI 已移除后纳入回归素材；去除该 UI 后旧实现仍只有 4 个内点。新增测试包含区域外强纹理干扰、不依赖区域外像素的特征提取、非零坐标偏移、无效区域拒绝，另用已有海岛地图三个位置和三种缩放作独立检验，并验证用户图的缩放、对比度、JPEG 压缩与错误地图交叉拒绝。最终共通过 74 项逻辑测试、13 项截图测试；这提供跨样例的回归证据，不代表所有游戏区域和动态遮挡条件都已实机验收。

运行路线、恢复和 4C 逻辑测试：

```powershell
.\.venv\Scripts\python.exe -u run_test.py tests.TestRouteMarkers tests.TestRouteMarkerSelection tests.TestRouteSteering tests.TestMinimapHeading tests.TestMinimapTracker tests.TestMapFeatureRegions tests.TestEchoRoute tests.TestFarm3CEchoTask tests.TestEchoFarmingRecovery tests.TestFarmEchoPickup tests.TestFarmEchoTeleport tests.TestFarmEchoTaskEchoRate tests.TestDeathReentry tests.TestBaseCombatTask tests.TestDomainRecoveryLoop
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

持续移动参考 `PathExecutor.MoveTo()` 的按住移动键、限频定位/纠偏和退出释放；队列传送参考 `HandleTeleportWaypoint()` 与 [TpTask](https://github.com/babalae/better-genshin-impact/blob/704d920d6d8fe09ae86628dab1e48464f0e4595e/BetterGenshinImpact/GameTask/AutoTrackPath/TpTask.cs) 的传送入口和传送后重定位。鸣潮实现使用自己的图鉴界面与录制地图，不兼容 BetterGI 的世界坐标和传送点数据库。

定位修复另复核 BetterGI 的 [SIFT 地图实现](https://github.com/babalae/better-genshin-impact/blob/b3f90e70494320639eeccc8c559612d02d2d9ef0/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/SceneBaseMap.cs)、[模板地图实现](https://github.com/babalae/better-genshin-impact/blob/b3f90e70494320639eeccc8c559612d02d2d9ef0/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/SceneBaseMapByTemplateMatch.cs) 与 [NavigationInstance](https://github.com/babalae/better-genshin-impact/blob/b3f90e70494320639eeccc8c559612d02d2d9ef0/BetterGenshinImpact/GameTask/AutoPathing/NavigationInstance.cs)。其现有模板模式采用多尺度、遮罩、局部搜索和置信度检查；并非仅有 SIFT 可选。本次提交修复鸣潮现有 SIFT 的有效区域提取，不引入原神模板/世界地图或另一套未经实机验证的匹配模式。特征限额先于遮罩的执行顺序可见 [OpenCV 官方 SIFT 源码](https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp)：`retainBest` 在 `runByPixelsMask` 之前。

[REA](https://github.com/morluto/rea/tree/500e32399a3bc827e17fb4adaa97db3bc0df29b5) 是可选的源码研究工具，运行 3C 任务不依赖它。开发机已配置 `rea-agents` 6.4.0 MCP 和配套 skill；这些电脑级配置不会随 Git 同步。本次检查使用源码和本地离线测试，未通过 REA 执行游戏目标操作，也未安装 Hopper/Ghidra/IDA 等分析引擎。
