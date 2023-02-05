# Assignment 2

## 环境
- Python 3.8.12
- PaddlePaddle 2.2.1 + CUDA 11.2

## 运行
直接运行 `main.py` 是在原始训练集和划分好的训练集上进行训练和验证一整个步骤，

运行 `tune.py` 是改进后的流程（如果没有 `ckpt/split.pdparams`，需要先运行 `main.py`得到模型参数）