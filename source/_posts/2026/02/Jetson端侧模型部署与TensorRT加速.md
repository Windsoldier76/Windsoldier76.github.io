---
title: Jetson端侧模型部署与TensorRT加速
date: 2026-02-14 20:19:06
tags:
  - Nvidia
  - TensorRT
  - ubuntu
  - jetson
  - 踩坑
category: 技术-记录
---

## 基于Jetson Orin Nano Super 8G版本的YOLO模型端侧部署与TensorRT加速推理

### 环境部署

首先通过Nvidia SDK Manager配置好基本的JetPack环境，本设备安装JetPack`6.2.1`，cuda`12.6.68`，cudnn`9.3.0.75`，TensorRT`10.3.0.30`

注意，如果前面安装正常但是jtop显示未检测到jetpack，那很有可能是版本问题导致的显示异常，只需要修改jtop的显示页面即可，jetpack通常应当是正确安装上了的。

#### 安装uv包

运行如下命令

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

下载脚本并执行，跳出来的选择直接回车，默认就是yes。

```bash
cd 项目目录
uv init
uv python install 3.10
uv python pin 3.10
uv venv --python 3.10 --system-site-packages
```

#### 配置tensorRT

首先在基础环境中运行

```bash
python3 - << 'EOF' 
import tensorrt as trt 
print(trt.__version__) 
EOF
```

如果输出结果是`10.3.0`或其他版本号，则说明TensorRT已正确安装进系统环境中，上一步中的`uv venv --python 3.10 --system-site-packages`，通过`--system-site-packages`将系统环境继承到了虚拟环境中，此时运行

```bash
uv run python3 - << 'EOF' 
import tensorrt as trt 
print(trt.__version__) 
EOF
```

应当输出结果与上一个命令一致。

#### 安装torch与ultralytics

打开`pyproject.toml`，编写如下内容

```text
[project]
name = "项目名称"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10, <3.11"
dependencies = [
    "numpy<2",
    "opencv-python",
    "ultralytics",
    "pytubefix>=10.3.6",
    "onnx>=1.12.0, <2.0.0",
    "onnxslim>=0.1.71"
]

[project.optional-dependencies]
jetson = [
    "torch",
    "torchvision",
    "onnxruntime-gpu",
]

# ================================
# uv：为特定包指定安装来源
# ================================

[tool.uv.sources]
torch = { url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl" }
torchvision = { url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl" }

onnxruntime-gpu = { url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl" }
```

更新完成后运行

```bash
uv sync --extra jetson
```

环境更新完成后运行

```bash
uv run python - << 'EOF'
import tensorrt as trt
import torch
print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__)
EOF
```

输出结果应当为

```bash
TensorRT: 10.3.0
Torch: 2.9.1
```

### 模型量化导出

```python
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
#results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
#results = model.val()

# Perform object detection on an image using the model
results = model("./bus.jpg", save=True)

# Export the model to ONNX format
success = model.export(format="engine", device=0, int8=True)
```

模型自动下载并导出到路径下`yolo11n.engine`

### 量化模型推理

```python
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.engine")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
#results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
#results = model.val()

# Perform object detection on an image using the model
results = model("./bus.jpg", save=True)

# Export the model to ONNX format
# success = model.export(format="engine", device=0, int8=True)
```

### 性能对比

来一张经典的公交车图片进行测试

```bash
# 使用TensorRT量化加速
image 1/1 ./bus.jpg: 640x640 4 persons, 1 bus, 10.9ms
Speed: 8.3ms preprocess, 10.9ms inference, 19.2ms postprocess per image at shape (1, 3, 640, 640)

# 基础的pt模型
image 1/1 ./bus.jpg: 640x480 4 persons, 1 bus, 162.1ms
Speed: 5.9ms preprocess, 162.1ms inference, 24.4ms postprocess per image at shape (1, 3, 640, 480)
```

性能差距确实夸张，下一步开始尝试将工作中的几个模型转换到tensorrt上试试。

有机会了再试试基于openvino的，这样就可以跑在英特尔的cpu上了。

至于国产的龙芯和晟腾还是再往后稍稍吧，一步一步来。
