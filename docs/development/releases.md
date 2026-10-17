# Windows 完整版与发布

## 下载使用

下载入口为 [vv910/better-ww Releases](https://github.com/vv910/better-ww/releases)。选择 `BetterWW-v版本号-windows-x64.zip`，完整解压后双击 `BetterWW.exe`。软件为 Windows x64 便携版，包含 Python 3.12、Qt、OCR 和声骸模型及运行库，不需要额外安装 Python。

启动失败会在软件目录写入 `startup-error.log`；运行日志和个人设置保存在同一目录下。程序需要能写入该目录。若游戏以管理员权限运行，软件也需要以管理员权限运行。

本 fork 不使用上游发布证书或更新仓库。更新时先退出旧版，再完整解压新版，将自己的 `configs`、`routes` 和需要保留的 `logs` 文件复制到新版对应位置。路线的 JSON 和 `.map.png` 必须一起保留。不要用 GitHub 自动生成的 Source code 包代替完整软件包。

## 本地构建

使用 Windows x64、仓库本地 Python 3.12 虚拟环境，并初始化子模块。安装运行与构建依赖后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
.\.venv\Scripts\python.exe scripts/build_portable.py --version v0.1.0
```

构建工具只从 Git 已跟踪的文件中暂存运行源码与资源；新增模块需先 `git add`。个人配置、日志、测试图片和模板训练子模块不会进入软件包。构建临时文件位于 `build/portable`，输出目录为 `dist/BetterWW`，同时生成 ZIP 与 `.sha256` 校验文件。

执行完整包自检：

```powershell
$app = Start-Process .\dist\BetterWW\BetterWW.exe -ArgumentList '--self-test' -Wait -PassThru -WindowStyle Hidden
if ($app.ExitCode -ne 0) { throw 'Portable self-test failed' }
Get-Content .\dist\BetterWW\smoke-test.json
```

自检在实际打包后的进程中检查主要导入、Qt 初始化、模型、翻译、路线资源和 OpenVINO 设备。发布前还应验证不依赖开发环境 PATH/PYTHONPATH 的独立启动；自检不等于游戏实机验收。

## GitHub 发布流程

`.github/workflows/fork-release.yml` 在推送 `v*` 标签时运行，使用 GitHub 自动提供的 `GITHUB_TOKEN`。流程分别启动测试模块、构建完整包、运行 EXE 自检，然后上传 ZIP 和 SHA256 到草稿 Release。确认产物后将草稿发布，下载链接才对其他人可见。

原有官方同步与签名工作流限定在上游仓库运行，fork 构建不会写入官方更新仓库或调用其签名服务。

首次版本为 `v0.1.0`；后续稳定版本递增 patch。标签使用 annotated tag，禁止覆盖已有标签。发布前检查远端标签与工作树，仅提交相关修改，并按 `.agents/skills/deploy/SKILL.md` 完成发布验证。

GitHub Release 应说明版本变化、下载文件、路线迁移方式以及尚未完成的实机验证；同时保留自动生成的同版本源代码入口。
