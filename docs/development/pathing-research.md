# BetterGI 路径追踪调研与 3C 导航改进

调研日期：2026-10-23。BetterGI 源码固定于默认分支 `refactor/20260930` 的提交 `887548a11ef183d8cd92c6ddf596776cebffb436`；本项目分析起点为 `018aead`。本文区分源码能力、离线验证和游戏验收。

## 结论

3C 导航可以继续改进。当前优先级是可靠定位、保存实际通行路线、统一地图坐标和身份，再扩展复杂地形动作。星标仅能表达目的地；星标相邻不意味着两者之间存在直线通路。

BetterGI 是 [GPL-3.0 开源项目](https://github.com/babalae/better-genshin-impact/tree/887548a11ef183d8cd92c6ddf596776cebffb436)。其地图追踪的主要能力来自地图数据、视觉定位、路线语义和闭环控制的组合，不能把运行效果归因于单一识别阈值。原神地图图像、坐标换算、传送点和动作参数不能直接用于鸣潮。

## 为什么 BetterGI 能跟得更稳

### 定位有独立的第二条路径

[SceneBaseMapByTemplateMatch.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/SceneBaseMapByTemplateMatch.cs) 实现模板粗搜索和精定位。它可以在没有 SIFT 几何提案时定位，并利用上一位置减少搜索范围。固定素材尺寸和缩放标定让它不必每帧盲搜所有比例；其中的相似度阈值适用于其素材和相关算法，不能直接照搬到鸣潮截图。

本项目 `MapLocalizer` 已有 SIFT、局部/全图搜索、图标遮罩、几何校验、短时光流和大地图恢复。但原有 `TerrainMatcher.verify()` 只能校验 SIFT 提供的变换：少于六个匹配点时没有变换，无法调用这条回退路径。光流也必须先有一次可靠的绝对定位。

### 持续观察实际运动并纠偏

[PathExecutor.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/AutoPathing/PathExecutor.cs) 按路线执行定位、朝向调整、移动和恢复，位置与动作形成反馈。本项目也已有持续按键、相机方向估计和位置窗口脱困；继续增加按键动作，不能替代定位或补出地图上的通路。

BetterGI 的普通移动与目标精确接近分开处理：移动期间持续计算到目标的方向，持续偏航时停步转向，接近目标后放慢并重新识别。[CameraOrientation.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/Common/Map/CameraOrientation.cs) 根据相机扇形估计朝向并检查置信度；[CameraRotateTask.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/AutoPathing/CameraRotateTask.cs) 包含转向增益和 DPI 补偿。这里没有发现自动学习鼠标增益的控制器，不能描述为自主学习。

### 底图与路线共享同一坐标系

[BaseMapLayerByTemplateMatch.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/BaseMapLayerByTemplateMatch.cs) 的楼层数据包含 `LayerGroupId/LayerId/Scale/Floor/Top/Left`；标准地图坐标由比例和偏移转换而来。[SceneBaseMap.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/SceneBaseMap.cs) 也提供图像坐标和共享地图坐标的正反变换。底图不是随某次游戏窗口截图改变的屏幕坐标。

[PathRecorder.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/AutoPathing/PathRecorder.cs) 先获得稳定定位，再转换并写入路线；识别失败不会写点。它同时保存地图名、匹配方式和版本。[Waypoint.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/AutoPathing/Model/Waypoint.cs) 区分传送、途经、目标和朝向点，并保存步行、疾跑、攀爬、飞行、游泳等动作。

实读 [脚本仓库](https://github.com/babalae/bettergi-scripts-list/tree/879bd43edcbceea4d13c20cede89f3d63c09ce52/repo/pathing) 的地下矿区水晶路线，23 个点包含传送、疾跑、战斗、起飞/落地、挖矿及按键脚本。作者已提供可通行路线和动作；执行器沿这些点持续纠偏。在本次检查的 AutoPathing 执行链中没有发现全图 A*、Dijkstra 或 NavMesh 规划。

### 恢复有状态和预算

[NavigationInstance.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/AutoPathing/NavigationInstance.cs) 的稳定定位接口会在局部匹配失败或明显跳变时尝试全局匹配。普通运行接口与稳定定位接口不同，不能假定所有调用自动获得同样的回退。

[TrapEscaper.cs](https://github.com/babalae/better-genshin-impact/blob/887548a11ef183d8cd92c6ddf596776cebffb436/BetterGenshinImpact/GameTask/AutoPathing/TrapEscaper.cs) 结合误攀爬等状态尝试后退、侧移、跳跃和偏转。执行器根据实际位移判停滞，限制恢复次数，并可重启当前传送段。这属于局部反应式恢复，不能据此推断程序已经识别了障碍物形状。

### 有数据条件和使用边界

[官方地图追踪说明](https://www.bettergi.com/feats/autos/pathing.html) 要求锁定小地图方向，也承认低纹理地区定位困难、路线点位和角色差异可能导致卡住。该页面与开发分支的更新节奏不同，不能用其中较早的分层限制概括最新源码，也不能由最新源码的分层数据结构推断所有地图都已实测。

## 当前失败证据及其限度

另一个开发会话中最近的运行日志实际加载 `local-route5.json`，起点为 `(1340, 752)`。失败发生在起点检查：小地图匹配点为 1 或 0，之后大地图重定位未确认。用户随后提供的 JSON 起点为 `(1388, 741)`，引用 `local-routeff2` 图组；两者不能作为同一次运行的数据合并分析。

仓库已有的两组真实地图/小地图素材，在本轮改动前均能定位：`3c-terrain-map.png` / `3c-terrain-minimap.png` 约为 `(1322, 691)`；`path.png` / `mini_map.png` 约为 `(1206, 469)`。这些素材适合验证独立算法、图像退化和错误地图拒绝，但不等于复现上述最新失败。完整闭环仍需在测试电脑执行并核对同一次运行的路线、图片和诊断日志。

## 独立模板搜索的实验结论

本轮在临时目录验证了全局多尺度模板搜索，没有将无约束的全局搜索加入运行时。不能直接复用原有 SIFT 提案校验的阈值：在真实用户地图上，全局灰度相关把约 `(1271, 970)` 的相似地形认成正确位置；真实位置约 `(1322, 691)`。错误候选的相关分约 0.56，峰间隔约 0.18，且四象限都得到支持，原有 verifier 仍可能放行。该校验原本依赖 SIFT 先约束几何位置。

进一步尝试去低频的 Laplacian 特征，在两张原始真实小地图上可找到正确位置，但用户图的缩放、模糊，以及独立素材的旋转测试仍暴露拒绝或误匹配。原型在两个 OpenCV 工作线程下耗时约 0.43–0.55 秒，不能由原始图成功推断泛化成功。

提高粗模板的最低有效尺寸可减少小块纹理假峰，但现有素材不足以证明全局定位的泛化。因此，无约束的独立搜索仍需可信的地图比例、素材预处理和更全面的负例。此次保留原有定位门槛；没有把“无 SIFT 匹配”变成降低置信门槛后继续移动。

## 2026-10-23 的恢复改进

`Farm3CEchoTask.recover_route_position()` 复用传送搜索的连续帧稳定判据。打开地图、点击居中后都等待地图停止平移；玩家箭头必须在三个新稳定视图中一致，首末两次地形配准也必须通过且路线坐标一致。地图观察受十秒总预算及单次三秒等待上限限制，超时配准不得生成提示；取消仍关闭地图并释放输入。单次 OpenCV 运算不能中途抢占，返回后会再次检查预算。

`TerrainSearch` 只在强配准产生的地图提示附近查找独立的小地图图像证据：

- 提示必须来自 `locate_map_player()` 的 `MapPosition(source='map')`，不能使用路线起点或普通坐标元组。
- 提示只保留五秒，每秒最多尝试一次，任何绝对定位成功后立即消费；光流不能延长提示有效期。
- 搜索中心限定在提示的 15 个录图像素内，使用降采样的 Laplacian 地形，在多个比例间比较候选；粗模板小于 48 像素的比例不受支持。
- 候选必须通过地形相关评分、不同位置/比例的候选竞争，再通过原有全图灰度唯一性和空间分区校验。最终位置仍须在提示附近，并遵守已有最大位置跳变限制。
- 明确的旋转、缩放、位置跳变或越界冲突不会被此搜索覆盖。结果标记为 `source='template'`，不伪造 SIFT 内点数。

这使恢复能在缺少 SIFT 匹配时尝试重新取得小地图位置；提示本身始终不能充当当前位置。回退只负责首次恢复定位，之后仍需正常绝对定位或原有三秒光流预算；不能在一直看不清地图时无限续走。现有路线格式和任务配置无需修改。

两组真实原始/缩放素材共六个子用例的独立搜索误差约 0.24–0.79 录图像素，开发机两个 OpenCV 工作线程下算法耗时约 125–203 ms，不包含截图和地图恢复。将提示故意偏移 `(8, -5)` 后仍能恢复到图像中的真实位置；错误地图、旋转、偏移过大的提示、重复地形和空白遮罩均有拒绝测试。集成测试另强制 SIFT 零匹配，验证恢复确实能使用独立地形证据。这些仍是有限素材上的离线结果，不是最新 `route5` 的实机成功率。

## 2026-10-24 的持续定位与到点控制

本轮复现并修复了三个问题：固定移动间隔可让高速角色反复跨过狭窄目标区；一次位置离群即可触发到点；识别计算期间已过期的光流仍可能被用于移动。

`RouteApproach` 根据观测到的地图位移和时间估计速度，在接近目标时使用更短的按键保持，并在下一次截图前松键，避免识别延迟继续把角色推过目标。途经点仍按顺序执行，不用直线穿越判据跳过拐点。最终点、战斗点、拾取点会先松键，等待输入释放，再请求绝对定位并读取新画面确认；纯光流结果不能触发这些点的完成。战斗、丢定位、脱困和地图恢复会清空速度估计。受限地图恢复与重新对齐的耗时不占步行预算，但不会增加其次数或地图观察预算。

`MapLocalizer` 在识别返回时重新检查时效：图像超过一秒不得作为当前位置，光流的绝对锚点与前帧间隔在计算结束后也须有效。朝向识别或转镜耗时让位置超过一秒时，任务停步重读。局部尝试中的明确几何冲突不会被后一次全图缺少匹配的原因覆盖。

持有路线移动键时，获取截图使用半秒等待预算；没有新图或返回耗时超过一秒就松键、清空定位并停止。此时仅在捕获/休眠作用域内暂停框架会绕过该预算的后台截图检查，退出或取消时恢复，战斗和静止阶段保留原检查。起步等静止阶段若截图超时返回空值，也会停止，避免通过 `frame` 属性隐式重试抓图后继续走。单次底层截图调用无法中途抢占，耗时超限会在返回时检查。

`TrackingVerifier` 为持续缺少 SIFT 特征的场景提供独立地形复核：有效光流只提出位置，最后一次绝对定位的缩放与旋转被冻结，再用原有全图唯一峰、地形分区和候选一致性校验取得新证据。复核通过才输出 `terrain_track` 并更新绝对锚点；失败时仍受原三秒光流预算限制。每秒最多尝试一次，不使用失效光流，不能通过相对跟踪自行延长定位寿命。

验证材料包含两组真实截图生成的平移序列、错误地图、旋转、重复地形、图标扰动、低质量光流，以及强制 SIFT 缺少匹配的连续六秒序列。控制仿真覆盖正负方向和斜向、150–300 录图像素/秒、50–130 ms 截图延迟、离群到点、恢复和取消。这些是离线验证，尚未取得新的连续游戏录像或整条路线实机结果。

## 自动失败复现包 {#navigation-replay}

每条当前路线在内存中保留最近最多 120 个小地图，原始图总量最多 24 MiB；移动时只复制小地图和事件，不编码图片或写盘。识别失败等异常退出时，先释放移动键，再尝试一次性保存到 `logs/3c-replays/<唯一编号>/`；用户主动取消不生成失败包，磁盘异常不会掩盖原始任务错误。

包内包含实际选中的路线快照、引用的主图与概览副本、可用的箭头模板、连续小地图和 JSONL 事件，记录定位来源、拒绝原因、耗时、地图提示/重置、朝向、目标和按键命令。地图副本在 `map_region` 外置零，原文件不修改，图像坐标保持不变。内存淘汰的帧、缺失图片和不完整资产都会明确报告，不能把截断记录当作完整实机录像。

现有“导出日志”会递归收集 `logs`，因此正常保存的复现包可随同导出，无需增加任务配置。离线工具只读取文件：

```powershell
.\.venv\Scripts\python.exe scripts/replay_3c_navigation.py "logs/3c-replays/替换为实际编号"
.\.venv\Scripts\python.exe scripts/replay_3c_navigation.py "logs/3c-replays/替换为实际编号" --relocalize
```

默认只统计定位可用率、来源、拒绝原因、失配间隔、已接受位置跳变和识别耗时；已接受的坐标不等于真实坐标。显式 `--relocalize` 才重新运行视觉，比较原始与重放统计。重放保留帧间隔和地图提示年龄，缺帧会中断跟踪；无法精确重现另一台电脑内部每一步计算的时钟，不能替代游戏内验收。

## 后续结构性改造

1. **标准比例下的全局重定位**：在本轮受限恢复基础上，使用标定底图与独立负例集继续研究无需大地图提示的定位，不能直接扩大全局搜索范围或放宽阈值。
2. **真实步行路线录制**：记录经定位验证的位置序列和拐点，区分普通途经点、战斗点和拾取点。压缩路线时保护拐角及动作点，避免为了少几个点而切过墙体、悬崖或道路弯角。
3. **地图身份与标准坐标**：建立鸣潮自己的地图底图与版本清单，标定截图到地图坐标的变换，保存明确的地图/层身份和传送点目录。已有的菜单文字和截图书签不足以证明地图身份。官方瓦片配准研究还不是经过游戏坐标验收的运行时地图系统。
4. **分段动作与恢复**：在可靠定位和实际通路基础上加入有边界的跳跃、攀爬、飞行等动作；卡住时返回可靠途经点、重定位或重新传送。恢复必须受次数和时间限制。
5. **实机验收**：用独立区域、不同截图尺寸和缩放、战后视角变化、低纹理地形及取消场景，记录起点成功率、定位误差、路线完成率和恢复次数。离线测试数量不能替代这些指标。

直接对大地图图片执行 A* 需要可信的通行区域、层间连接和高度信息；当前并没有这些输入。仅凭平面地图颜色推断可通行性，无法可靠处理鸣潮中的高低差、建筑和悬崖。

## 复核命令

2026-10-24 最终定向回归：212 项逻辑测试和 16 项真实截图测试通过，文档严格构建通过。未运行游戏内完整路线验收。

2026-10-23 最终定向回归：154 项逻辑测试和 16 项真实截图测试通过，文档严格构建通过。未运行游戏内完整路线验收。

使用仓库 `.venv`，逻辑回归命令如下；截图模块必须独立启动进程。

```powershell
.\.venv\Scripts\python.exe -u run_test.py tests.TestMapHintSearch tests.TestTerrainSearch tests.TestMinimapTracker tests.TestTerrainMatcher tests.TestMapFeatureRegions tests.TestEchoRoute tests.Test3CRouteRuntime tests.TestRouteMapNavigation tests.TestFarm3CEchoTask
.\.venv\Scripts\python.exe -u run_test.py tests.TestLocalizationFreshness tests.TestContinuousTerrainTracking tests.TestTrackingVerifier tests.TestRouteCapture tests.TestRouteSteering tests.TestNavigationReplay tests.TestNavigationReplayTask
.\.venv\Scripts\python.exe -u run_test.py tests.Test3CRouteImages
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

本轮同时更新截图诊断测试的旧断言：上一提交已保存 `last-attempt-minimap`，旧测试的预期列表尚未包含它；现在也核对保存的图片确实等于最后一次送入算法的小地图。
