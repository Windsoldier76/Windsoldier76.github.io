---
title: tmux常用命令
date: 2026-03-03 22:16:30
tags: 
  - tmux
  - linux
  - 常用命令
category: 技术-记录
comment: true
---

内容搬运自[菜鸟教程](https://www.runoob.com/linux/linux-comm-tmux.html)

# 常用命令与快捷键

tmux 的所有操作都需要先按下前缀键（默认是 Ctrl+b），然后输入命令键。

## 会话管理

 | 命令/快捷键            | 说明                           |
 |-----------------------|-------------------------------| 
 | tmux new -s <name>    | 创建名为 name 的新会话         | 
 | Ctrl+b d              | 分离当前会话（会话继续后台运行） | 
 | tmux ls               | 列出所有会话                   | 
 | tmux attach -t <name> | 重新连接到指定会话              | 
 | Ctrl+b $              | 重命名当前会话                 | 
 | Ctrl+b s              | 切换会话                       | 
## 窗口管理

 | 命令/快捷键      | 说明                |
 |-----------------|--------------------| 
 | Ctrl+b c        | 创建新窗口          | 
 | Ctrl+b &        | 关闭当前窗口        | 
 | Ctrl+b n        | 切换到下一个窗口     | 
 | Ctrl+b p        | 切换到上一个窗口     | 
 | Ctrl+b <number> | 切换到指定编号的窗口 | 
 | Ctrl+b ,        | 重命名当前窗口       | 
## 面板管理
 | 命令/快捷键     | 说明               |
 |----------------|--------------------| 
 | Ctrl+b %       | 垂直分割当前面板    | 
 | Ctrl+b "       | 水平分割当前面板    | 
 | Ctrl+b <arrow> | 在面板间移动焦点    | 
 | Ctrl+b x       | 关闭当前面板        | 
 | Ctrl+b z       | 最大化/恢复当前面板 | 
 | Ctrl+b Space   | 切换面板布局        | 