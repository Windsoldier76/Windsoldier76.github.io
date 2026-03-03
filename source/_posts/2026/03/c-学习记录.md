---
title: c++学习记录
date: 2026-03-03 22:18:18
tags: 
  - c++
  - vscode
  - msys
category: 技术-记录
comment: true
---

# 20260303 学习配置vscode的c/c++运行环境

安装参考:

`https://blog.csdn.net/qq_42417071/article/details/137438374`

`https://zhuanlan.zhihu.com/p/1906303263369852149`

## 安装msys2

进入[官方github release页面](https://github.com/msys2/msys2-installer/releases/)
下载[`msys2-x86_64-20251213.exe`](https://github.com/msys2/msys2-installer/releases/download/2025-12-13/msys2-x86_64-20251213.exe)

在安装完成的页面中选中运行msys，或者打开`msys2 ucrt64`(通过windows自带的搜索就可以找到)

在终端中输入

```bash
pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain
```

当系统提示是否继续安装时，请输入y并回车。

打开安装 `MSYS2` 的目录，先找到`ucrt64`文件夹并进入，再找到`bin`文件夹并进入，然后在地址栏中，复制路径。

如果一开始用默认路径，那路径就是`C:\msys64\ucrt64\bin`

将这个路径添加到环境变量`PATH`中，打开终端测试是否安装成功

```cmd
gcc --version
g++ --version
gdb --version
```

## VScode和C/C++插件下载安装

这一步随便找教程就搞定了，跳过

## 调试c程序

点击调试时弹窗选择`c/c++:gcc.exe 生成和调试活动文件`,有`code runner`插件也可以使用`F1`或者右键选中`Run Code`，
如果出现弹窗`c/c++扩展的预发行版本可用`，选择`是`

## 常见问题排查

最好安装**code runner**插件，可以省去很多麻烦

中间遇到报错，无法通过vscode正确使用编译，报错内容为gcc无法定位程序输入点clock_gettime64于动态链接库文件，实际原因出现在环境变量中，参考 `https://blog.csdn.net/aaalifu/article/details/114436662` 和 `https://blog.csdn.net/weixin_43935899/article/details/131344343` 两个文章，查看环境变量发现`conda/mingw-w64`在前面，怀疑问题出在这里，将`msys2/ucrt64/bin`上移到上面之后确定保存并重启vscode，运行成功

