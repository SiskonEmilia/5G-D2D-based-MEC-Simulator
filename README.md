# 基于 5G D2D 的移动边缘计算仿真实验

## 环境要求

您需要在有 Python3 环境的电脑上运行本项目代码，且至少需要安装 numpy 库。

如果您想运行包括测试在内的代码，您还需要安装 jupyter 和 matplotlib 库。

本项目在 python 3.7.3 下通过运行测试，并不能保证在其他任何版本下有相同的测试结果。

## 项目结构及使用说明

项目目录结构如下图：

```plain
- 仿真代码.ipynb
- main.py
- README.md
- resultAlgorithm.png
- resultVValue.png
```

其中，`main.py` 内包含了本仿真实验的所有模拟代码，通过使用其中定义的类，你可以重现我们论文中所进行的实验，或者以其他参数进行测试，**其中只定义和实现了实验所需的各种测试类，直接使用 python3 运行本文件不会执行测试**。`仿真代码.ipynb` 包含了 `main.py` 的所有内容，并且使用论文中的设置参数进行了实验，您可以从中了解如何使用本项目中的代码，在 `jupyter notebook` 中运行本文件将会以论文参数执行一个较小规模的测试。`resultAlgorithm.png` 和 `resultVValue.png` 分别为各算法性能测试和 Thompson Sampling 不同 v 值下的性能测试的示意图，当你直接运行 `仿真代码.ipynb` 中的代码时，新的结果将被生成并保存在这两个文件中。
