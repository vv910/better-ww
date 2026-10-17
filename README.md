# Better WW · 3C 声骸路线版

基于 [ok-wuthering-waves](https://github.com/ok-oldking/ok-wuthering-waves) 与 ok-script 开发的鸣潮自动化分支。增加局部地图路线录制、3C 路线刷取、路线队列与信标传送，保留原有自动战斗和声骸拾取。

[下载 Windows 完整版](https://github.com/vv910/better-ww/releases) · [路线使用说明](docs/development/echo-farming.md) · [源码安装](docs/development/testing-on-another-pc.md) · [构建与发布](docs/development/releases.md)

## 下载并启动

1. 在本仓库的 Releases 页面下载 `BetterWW-v版本号-windows-x64.zip`。
2. 完整解压到一个可写目录。不要只拖出 EXE，也不要在压缩包里运行。
3. 双击 `BetterWW.exe`。游戏以管理员身份运行时，软件也需使用相同权限。
4. 选择鸣潮 PC 窗口和截图方式，进入任务列表。

完整包包含 Python、界面组件、OCR 模型、声骸模型和运行依赖，无需安装 Python、Git、REA 或另行执行 pip。GitHub 自动生成的 Source code 压缩包是开发源码，不是可直接运行的软件。

此分支的便携包由本仓库构建，未使用上游签名证书；更新时从本仓库 Releases 下载新版。退出软件后更新，保留个人 `configs`、`routes` 和日志。详见[发布说明](docs/development/releases.md)。

## 3C 路线刷取

1. 游戏小地图设为北向固定，开启镜头重置；确认软件中的游戏热键与实际一致。
2. 在大地图给起点放一个菱形，沿可通行路径至少放两个星标。
3. 执行“从地图录制 3C 路线”，检查生成的 `.preview.png` 和 JSON 点序。
4. 转弯、经过位置使用 `walk`；怪物点使用 `combat`；专门拾取点使用 `pickup`。
5. 将角色放在起点附近，关闭地图，执行“按路线刷取 3C 声骸”。

路线由 JSON 和配套 `.map.png` 共同组成，分享或迁移时两者都需要。完整包附带录制工具与说明，不附带已验证的成品刷怪路线。

分散路线可通过“路线队列”连续执行；录制时填写附近固定 Boss 的图鉴名称，并把起点放在已解锁信标附近。[传送配置与限制](docs/development/echo-farming.md#route-queue)。

默认开启：

- **保持视角回正**：行走时定期把视角放回角色身后；转弯后重校准，直线行走保持前进键。
- **离地自动开翼**：出发前自动选择控物工具，复用控物图标灰化的离地判断；连续确认离地后发送一次跳跃键，稳定落地后重新启用下一次保护。运行时不要手动反复按跳跃键。

离地保护是视觉触发的开翼尝试，不是地形高度规划；不保证极短落差、体力不足或游戏禁止开翼时能成功。仍应录制可通行的路线，绕开无落地点、深水和复杂攀爬段。软件没有怪物 COST 分类模型，3C 类型取决于你记录的怪物点。

## 故障反馈

请通过[本仓库 Issues](https://github.com/vv910/better-ww/issues)反馈版本、分辨率、队伍、任务日志和失败时间。定位失败时保留对应 `3c-location-*` 图片，并提供路线 JSON 与配套地图。启动失败时检查软件目录中的 `startup-error.log`。

开发与测试命令见[贡献指南](docs/development/contributing.md)。每个截图测试模块需通过 `run_test.py` 单独启动 Python 进程。

## 来源与许可

上游项目：[ok-oldking/ok-wuthering-waves](https://github.com/ok-oldking/ok-wuthering-waves)，框架：[ok-script](https://ok-script.com)。路径执行和视觉定位的研究参考 [BetterGI](https://github.com/babalae/better-genshin-impact)。本仓库路线和鸣潮地图不兼容原神脚本。

许可见 [LICENSE.txt](LICENSE.txt)。软件开源、免费，通过图像识别与常规输入进行操作，不读取游戏内存或修改游戏文件。使用第三方自动化仍需自行了解游戏运营规则。
