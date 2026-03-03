---
title: 在树莓派中部署clash
date: 2026-03-03 18:39:41
tags: 
  - 树莓派
  - clash
category: 技术-记录
comment: true
---

# 在树莓派5/4G版本上部署clash

>[参考链接1，树莓派4B_linux clash部署教程（2024.3.4）](https://github.com/Xizhe-Hao/Clash-for-RaspberryPi-4B)
>
>[参考链接2，在 Raspberry Pi 上运行 Clash 作为透明代理](https://cherysunzhang.com/2020/05/deploy-clash-as-transparent-proxy-on-raspberry-pi/)
>
>[参考链接3，树莓派（无桌面）运行 clash 代理服务](https://www.bilibili.com/read/cv44730464/?opus_fallback=1)

## 确认系统架构

```bash
uname -m
```

返回的型号应当是`aarch64`, 因此选择`mihomo-arm64`版本进行下载

如果不希望使用TUN模式的话也可以下载使用老的[`clash-armv8`](https://github.com/frainzy1477/clash_dev/releases/download/v1.1.0/clash-linux-armv8.gz)内核

## 下载mihomo内核(即clash-meta内核)

选择此内核的原因是希望能开启TUN模式

通过[官方github release页面](https://github.com/MetaCubeX/mihomo/releases)下载 [`mihomo-linux-arm64-v1.19.20.gz`](https://github.com/MetaCubeX/mihomo/releases/download/v1.19.20/mihomo-linux-arm64-v1.19.20.gz),注意是`arm64`的`gz`包

```bash
gunzip mihomo-linux-arm64-v1.19.20.gz

sudo mv mihomo-linux-arm64-v1.19.20 /usr/local/bin/clash
sudo chmod +x /usr/local/bin/clash
```

## 下载Country.mmdb

Country.mmdb 地理位置数据库文件，可以从[官方github release页面](https://github.com/Dreamacro/maxmind-geoip/releases) 下载 

## 下载config.yaml

自行寻找，注意名字需要调整为config.yaml，因为这是默认加载的配置文件，如果是别的名字的话请参考后面的[**注册为后台服务**](#注册为后台服务)章节

## 将配置文件放入文件夹

```bash
sudo mkdir -p ~/.config/clash/
sudo mv config.yaml ~/.config/clash/
sudo mv Country.mmdb ~/.config/clash/
```

## 运行

### 确认版本号

```bash
/usr/local/bin/clash -v
```

输出应当类似`Clash Meta v1.xx.x`

### 启动

```bash
clash -d ~/.config/clash
```

**注意**：`ctrl+c`不能关闭clash,需要额外输入关闭命令以正确停止clash运行

```bash
pkill clash
sleep 1
```

## 注册为后台服务

可以实现自动重启、开机自启

### 创建服务

```bash
sudo vim /etc/systemd/system/clash.service
```


### 编写服务文件

```service
[Unit]
Description=Clash Proxy Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=myusername
Group=myusername

ExecStart=/usr/local/bin/clash -d /home/myusername/.config/clash -f /home/myusername/.config/clash/config.yaml
Restart=on-failure
RestartSec=5s

# ===== TUN 必须的权限 =====
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_BIND_SERVICE
NoNewPrivileges=true

LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

**注意**：需要根据你的用户名替换`myusername`，clash的config.yaml可以根据你的配置文件名进行修改

### 启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable clash
sudo systemctl start clash
```

## 增加TUN模式

在`config.yaml`中增加一下内容

```yaml
# ===== TUN 模式 =====
tun:
  enable: true
  stack: system # 树莓派 5 + 新内核，推荐 system
  dns-hijack:
    - any:53
  auto-route: true
  auto-detect-interface: true

# ===== DNS（非常重要）=====
dns:
  enable: true
  listen: 0.0.0.0:1053
  enhanced-mode: fake-ip
  fake-ip-range: 198.18.0.1/16
  use-hosts: true

  nameserver:
    - 223.5.5.5
    - 119.29.29.29

  fallback:
    - 8.8.8.8
    - 1.1.1.1
    - tls://[2606:4700:4700::1111]:853
    - tls://[2001:4860:4860::8888]:853

  fallback-filter:
    geoip: true
    geoip-code: CN
    ipcidr:
      - 240.0.0.0/4
```

```bash
ip a | grep 198.18 
# 运行clash时返回以下结果为正常，常规检测ip a | grep tun是看不到的
inet 198.18.0.1/30 brd 198.18.0.3 scope global Meta
```

## 常用的config.yaml配置

```yaml
ipv6: true
allow-lan: true
external-controller: 0.0.0.0:9090
secret: 自行填写密钥，注意双引号也会被作为密钥使用
```

## 定期更新订阅

`config.yaml` 里加入如下内容

```yaml
# 顶级字段
profile:
  store-selected: true        # 记住你在 UI 里选的节点
  store-fake-ip: true         # 保留 fake-ip，重启不抖
  interval: 86400             # 自动更新间隔（秒），1 天，每小时更新的话改为3600

# 顶级字段
proxy-providers:
  airport:
    type: http
    url: https://你的订阅地址
    interval: 86400          # 每小时更新的话改为3600
    path: ./providers/airport.yaml
    health-check:
      enable: true
      url: http://www.gstatic.com/generate_204
      interval: 300

# 在proxy-groups里增加use字段，指定airport，此处为示例
proxy-groups:
  - name: 🔰 选择节点
    type: select
    use:
      - airport

```

## 部署基于本地的web ui

### yacd，简单稳定的web ui

#### 下载yacd文件

```bash
cd ~/.config/clash
mkdir -p yacd
cd yacd

# 下载官方编译好的单文件版本
wget https://github.com/haishanh/yacd/releases/latest/download/yacd.tar.xz
tar -xf yacd.tar.xz

ls
# index.html  assets/  ...
```

#### 配置yacd界面

```bash
vim ~/.config/clash/config.yaml
```

打开`config.yaml`并增加以下内容

```yaml
external-ui: yacd
```

重启clash，查看页面`树莓派ip地址:9090`

```cmd
sudo systemctl restart clash
```

### metacudexd，功能更多的web ui

#### 创建目录（严格对齐）

```bash
mkdir -p ~/.config/clash/meta-dashboard
cd ~/.config/clash/meta-dashboard
```

#### 下载官方构建好的 UI（不是源码）

```bash
wget https://github.com/MetaCubeX/metacubexd/releases/latest/download/metacubexd.tar.gz
tar -xzf metacubexd.tar.gz
```

解压后你应该看到：

```bash
index.html
assets/
```

确认路径最终是：

```bash
~/.config/clash/meta-dashboard/index.html
```

#### 配置metacubexd页面

```bash
vim ~/.config/clash/config.yaml
```

打开`config.yaml`并增加以下内容

```yaml
external-ui: meta-dashboard
```

重启clash，查看页面`树莓派ip地址:9090`

```cmd
sudo systemctl restart clash
```
