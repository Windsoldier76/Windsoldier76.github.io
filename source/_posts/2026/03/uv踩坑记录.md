---
title: uv踩坑记录
date: 2026-03-03 22:18:08
tags: 
  - python
  - uv
category: 技术-记录
comment: true
---

# 安装

### windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

指定版本

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.7.4/install.ps1 | iex"
```

### linux or macOS

curl方法

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

wget方法

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

指定版本

```bash
curl -LsSf https://astral.sh/uv/0.7.4/install.sh | sh
```


# 常用uv操作

### 项目初始化

```cmd

uv init

uv python pin 3.12

uv sync

uv add 第三方库
```

# 让uv走clash代理

根据[官方文档](https://uv.doczh.com/configuration/files/)中的内容:

> uv 还会在 macOS 和 Linux 系统上的 ~/.config/uv/uv.toml（或 $XDG_CONFIG_HOME/uv/uv.toml），或 Windows 系统上的 %APPDATA%\uv\uv.toml 中查找用户级配置；在 macOS 和 Linux 系统上的 /etc/uv/uv.toml（或 $XDG_CONFIG_DIRS/uv/uv.toml），或 Windows 系统上的 %SYSTEMDRIVE%\ProgramData\uv\uv.toml 中查找系统级配置。
>
> 用户级和系统级配置必须使用 uv.toml 格式，而不是 pyproject.toml 格式，因为 pyproject.toml 旨在定义一个 Python 项目。
>
> 如果同时找到项目级、用户级和系统级配置文件，设置将被合并，项目级配置优先于用户级配置，用户级配置优先于系统级配置。（如果找到多个系统级配置文件，例如同时在 /etc/uv/uv.toml 和 $XDG_CONFIG_DIRS/uv/uv.toml 中找到，将仅使用第一个找到的文件，XDG 优先。）
>
> 例如，如果项目级和用户级配置表中都存在一个字符串、数字或布尔值，将使用项目级的值，忽略用户级的值。如果两个表中都存在一个数组，数组将被连接起来，项目级设置在合并后的数组中靠前。
>
> 通过环境变量提供的设置优先于持久化配置，通过命令行提供的设置优先于环境变量和持久化配置。


在 `C:\Users\用户名\AppData\Roaming\uv\uv.toml` 下编辑

```toml
# uv 全局配置（Windows）
http-proxy  = "http://127.0.0.1:7890"
https-proxy = "http://127.0.0.1:7890"
# 如果你用 socks5，可选
# no-proxy = "localhost,127.0.0.1"
```

# uv设置为国内镜像

参考文章 [官方文档中的译者经验](https://uv.oaix.tech/blog/2025/06/17/quickly-set-uv-package-index-is-china-mirror/#3)

在项目的`pyproject.toml`中增加如下内容，注意以下源均不支持torch的gpu版本下载

科大源
```toml
[[tool.uv.index]]
name = "ustc"
url = "https://pypi.mirrors.ustc.edu.cn/simple/"
```

清华源
```toml
[[tool.uv.index]]
name = "tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
```

腾讯云源
```toml
[[tool.uv.index]]
name = "tencent"
url = "https://mirrors.cloud.tencent.com/pypi/simple/"
#url = "https://mirrors.tencentyun.com/pypi/simple/" # 内网
```

阿里云源
```toml
[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
#url = "http://mirrors.cloud.aliyuncs.com/pypi/simple/" # 内网
```

华为云源
```toml
[[tool.uv.index]]
name = "huaweicloud"
url = "https://mirrors.huaweicloud.com/repository/pypi/simple/"
```

火山引擎源
```toml
[[tool.uv.index]]
name = "volces"
url = "https://mirrors.volces.com/pypi/simple/"
#url = "https://mirrors.ivolces.com/pypi/simple/" # 内网
```

# 常用uv项目文件配置

### 带pytorch-gpu和清华源的配置文件

在`pyproject.toml`末尾增加

```toml
# 1. 配置清华源为“默认”源 (default = true)
#    uv 会优先在这里查找普通包 (如 numpy, requests 等)
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

# 2. 显式把官方 PyPI 加回来 (可选)
#    如果在清华源找不到包，uv 会尝试在这里查找
[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"

# 3. 配置 PyTorch CUDA 12.6 专用源
#    explicit = true 表示只有在 [tool.uv.sources] 指定了的包才用这个源，
#    避免 uv 去这个源里找无关的包从而拖慢速度。
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

# 4. 绑定具体的包到具体的源
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }
```

### flash-attn预编译版本

可以根据网站 `https://flashattn.dev/` 找到需要的版本所需的预编译wheel，这样就不需要单独编译了，在uv中也不会出现报错(这个问题主要是windows平台)

```toml
dependencies = [
    # 👇 直接指定 wheel（关键）
    "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu126torch2.10-cp312-cp312-win_amd64.whl",
]
```

# uv项目文件离线迁移

[参考链接](https://blog.csdn.net/weixin_41544125/article/details/155065054)

### uv离线安装包准备

通过[官方github release页面](https://github.com/astral-sh/uv/releases)下载二进制包

其中windows得到的是可以直接使用的exe文件，可以将其解压后放入到环境变量`PATH`中之后可以通过`uv --version`确认。
或者也可以直接在命令行中使用`文件根目录\uv.exe --version`进行操作，其中`.exe`可以不写。

### 环境wheel包导出
在源机器中运行

```cmd
uv lock
uv export --format requirements.txt > requirements.txt
```

然后下载文件

```cmd
pip download -r requirements.txt -d wheelhouse
```

当需要下载指定cuda版本的torch-gpu时使用`--extra-index-url`参数，如下：

```cmd
pip download -r requirements.txt -d wheelhouse --extra-index-url https://download.pytorch.org/whl/cu126
```

如果是从ubuntu等linux系统迁移则需要以下命令，此命令参考了[https://segmentfault.com/a/1190000047450089](https://segmentfault.com/a/1190000047450089)，需要根据情况进行修改:

```bash
# 3.下载所有依赖的离线安装包
pip download -r requirements.txt -d ./win_amd64 --python-version 3.12 --platform win_amd64 --only-binary=:all:
```

此时源机器的项目路径下会出现一个新的文件夹`wheelhouse`，里面包含了环境中所有的whl包，将这个文件夹和`requirements.txt`拷贝到目标机器中

### python离线安装(此部分未进行测试，仅搬运网络内容)

> [参考链接1](https://juejin.cn/post/7516571684254793728)
>
> [参考链接2](https://zhuanlan.zhihu.com/p/1918311401614704904)

##### 安装python

1. 使用官方网站下载

进入[python官方网站](https://www.python.org/downloads/)，注意下载standalone版本


2. 使用github release页面获取并安装

通过[python-build-standalone releases](https://github.com/astral-sh/python-build-standalone/releases)
页面获取python离线安装文件

windows下解压到目录后将其加入环境变量`PATH`，建议将其上移到最上面

完成后输入以下命令确认安装是否成功

```cmd
python3 --version
python --version
uv python list --no-managed-python
```

linux下运行如下命令

```bash
# 创建 Python 安装目录
sudo mkdir -p /usr/local/python-build-standalone

# 解压 Python
tar -xzf cpython-3.11.13+20250612-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz -C /usr/local/python-build-standalone

# 验证安装
/usr/local/python-build-standalone/bin/python3 --version

# 将 Python 添加到 PATH（优先级最高）
export PATH="/usr/local/python-build-standalone/bin:$PATH"

# 验证 UV 能够识别 Python
uv python list --no-managed-python
# 应该显示我们安装的 Python 路径
```

### 目标机器部署


目标机器中应当至少包含以下内容:
 - python代码
 - pyproject.toml
 - .python-version
 - requirements.txt

```cmd
uv venv
uv pip install --no-index --find-links wheelhouse -r requirements.txt
```

等待运行完成后，运行python文件测试

```cmd
uv python main.py
```