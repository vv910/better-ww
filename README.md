# Better WW · 3C 声骸路线版

基于 [ok-wuthering-waves](https://github.com/ok-oldking/ok-wuthering-waves) 与 ok-script 开发的鸣潮自动化分支。增加局部地图路线录制、3C 路线刷取、路线队列与信标传送，保留原有自动战斗和声骸拾取。

[下载 Windows 完整版](https://github.com/vv910/better-ww/releases) · [路线使用说明](docs/development/echo-farming.md) · [源码安装](docs/development/testing-on-another-pc.md) · [构建与发布](docs/development/releases.md)

本轮导航修复请使用 [v0.1.1-beta.1 测试版](https://github.com/vv910/better-ww/releases/tag/v0.1.1-beta.1)。稳定版 v0.1.0 不包含这些改动。

## 下载并启动

1. 在本仓库的 [Releases 页面](https://github.com/vv910/better-ww/releases)下载 `BetterWW-v版本号-windows-x64.zip`；测试版也会从此页面提供，请按发布说明确认版本类型。
2. 完整解压到英文可写目录，例如 `E:\BetterWW`；完整安装路径不能含中文（框架会拒绝启动）。不要只拖出 EXE，也不要在压缩包里运行。
3. 双击 `BetterWW.exe`。游戏以管理员身份运行时，软件也需使用相同权限。
4. 选择鸣潮 PC 窗口和截图方式，进入任务列表。

完整包包含 Python、界面组件、OCR 模型、声骸模型和运行依赖，无需安装 Python、Git、REA 或另行执行 pip。GitHub 自动生成的 Source code 压缩包是开发源码，不是可直接运行的软件。

此分支的便携包由本仓库构建，未使用上游签名证书；更新时从本仓库 Releases 下载新版。退出软件后更新，保留个人 `configs`、`routes` 和日志。详见[发布说明](docs/development/releases.md)。

## 3C 路线刷取

1. 游戏小地图设为北向固定，开启镜头重置；确认软件中的游戏热键与实际一致。
2. 打开大地图，把目标星标组放在中央附近。默认“Nearest Beacon”无需菱形：程序选取中央星标组，并自动选附近可见信标作为路线入口。首个信标到星标只受录制半径限制，可能跨过障碍；请检查预览并在需要时补途经星。Nearest Beacon 至少需要一个星标；`Diamond` 旧方式仍要求至少两个星标。
3. 执行“从地图录制 3C 路线”，检查生成的 `.preview.png` 和 JSON 点序。默认会逐步缩小地图，捕获最多 6 张配准通过的传送概览图；主路线坐标不变。
4. 转弯、经过位置使用 `walk`；怪物点使用 `combat`；专门拾取点使用 `pickup`。
5. 启动“按路线刷取 3C 声骸”。有书签的路线会尝试从当前世界地图搜索路线信标并传送，不要求角色先站在起点附近。

路线 JSON、`.map.png`、`.preview.png` 及生成的 `.view-N.png` 共同组成一套文件；分享或迁移时请完整保留。完整包附带录制工具与说明，不附带已验证的成品刷怪路线。

书签搜索只覆盖已录制的当前地图范围和同一地图层，不保证跨独立地图、未知层或所有视野都成功；旧路线填写的 Boss 图鉴锚点仍兼容。梦枢天罗等跨层路线本版暂不支持自动切层。[传送配置与限制](docs/development/echo-farming.md#route-queue)。

默认开启：

- **保持视角回正**：行走时定期把视角放回角色身后；转弯后重校准，直线行走保持前进键。
- **离地自动开翼**：复用控物图标灰化的离地判断；确认离地后尝试发送一次跳跃键。默认不会自动切换工具；若启动时没有选择控物，会提示本路线未启用防坠保护并继续刷取。可显式开启自动选择控物，准备失败时同样降级继续。

保持视角回正默认开启。离地保护是视觉触发的开翼尝试，不是地形高度规划；不保证极短落差、体力不足或游戏禁止开翼时能成功。仍应录制可通行的路线，绕开无落地点、深水和复杂攀爬段。软件没有怪物 COST 分类模型，3C 类型取决于你记录的怪物点。

导航用 8 秒实际位置窗口判断是否真正停住；路线绕障时到目标的距离暂时增加不会单独触发脱困。持续移动 16 秒仍未接近目标时会停步并尝试一次回正/重定位，再观察 30 秒；确认真卡住才执行有限的短撤退或侧移恢复，并在动作间检查空中与敌人状态。脱困跳跃只在控物地面信号可用时尝试，信号不可用时跳过跳跃。

路线书签采集、有限视图搜索、同层地图配准和小地图恢复目前只有离线验证，尚待测试电脑实际游戏验收。`TerrainMask` 会统一排除地图文字/星标、紧凑蓝色信标和金色箭头等 UI 干扰，不降低定位门槛；这不代表所有地图误识别均已解决。梦枢预览中的一处假星标目前无法用现有叠加预览复现，录制时仍需人工核对点序；`map_layer` 仅为预留字段，本版不会自动切换地图层。

## 故障反馈

请通过[本仓库 Issues](https://github.com/vv910/better-ww/issues)反馈版本、分辨率、队伍、任务日志和失败时间。定位失败时保留对应 `3c-location-*` 图片，并提供路线 JSON 与配套地图。启动失败时检查软件目录中的 `startup-error.log`。

开发与测试命令见[贡献指南](docs/development/contributing.md)。每个截图测试模块需通过 `run_test.py` 单独启动 Python 进程。

## 来源与许可

上游项目：[ok-oldking/ok-wuthering-waves](https://github.com/ok-oldking/ok-wuthering-waves)，框架：[ok-script](https://ok-script.com)。路径执行和视觉定位的研究参考 [BetterGI](https://github.com/babalae/better-genshin-impact)。本仓库路线和鸣潮地图不兼容原神脚本。

许可见 [LICENSE.txt](LICENSE.txt)。软件开源、免费，通过图像识别与常规输入进行操作，不读取游戏内存或修改游戏文件。使用第三方自动化仍需自行了解游戏运营规则。
