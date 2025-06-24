# FashionMNIST分类项目 - 基于LeNet网络

该项目使用简化版的LeNet网络对FashionMNIST数据集进行分类。

## 项目结构
FashionMNIST-LeNet/
├── data/ # 自动下载的数据集（不需要上传）
├── models/ # 保存训练好的模型
├── results/ # 保存训练结果图像
├── src/ # 源代码目录
│ ├── dataset.py # 数据集加载和处理
│ ├── model.py # 模型定义
│ ├── train.py # 训练脚本
│ └── utils.py # 工具函数
├── README.md # 项目说明文档
└── requirements.txt # 依赖库列表

## 如何运行
1. 安装依赖：
pip install -r requirements.txt

2. 运行训练：
cd src
python train.py

## 参数调整
可以通过修改`train.py`中的参数来调整训练：
- `epochs`: 训练轮数（默认10）
- `batch_size`: 批次大小（默认128）

## 结果
训练完成后，可以在`results/`目录查看训练曲线，在`models/`目录找到最佳模型。
