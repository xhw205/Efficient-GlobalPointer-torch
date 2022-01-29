## Efficient GlobalPointer：少点参数，多点效果

### 介绍

基于 GlobalPointer 的改进，[Keras 版本](https://spaces.ac.cn/archives/8877) 的 torch 复现，核心还是 token-pair 。

绝大部分代码源自本人之前关于 GlobalPointer 的 [repository](https://github.com/xhw205/GlobalPointer_torch)。

> 笔者已经将 GlobalPointer 落地部署，垂直领域特别是嵌套情况下的信息抽取，GP真的做的很好，现在 Efficient GP 参数更少，效果更优，所以我认为只要上了 BERT，就抛弃 BIO-CRF 那一套吧！

### 数据集

[CMeEE](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414) 存放在:
train_cme_path = ./datasets/CME/train.json
eval_cme_path = ./datasets/CME/dev.json

### 预训练模型

1、笔者比较喜欢用RoBerta系列 [RoBERTa-zh-Large-PyTorch](https://github.com/brightmart/roberta_zh)

2、点这里直接[goole drive](https://drive.google.com/file/d/1yK_P8VhWZtdgzaG0gJ3zUGOKWODitKXZ/view)下载

### 运行

注意把train/predict文件中的 bert_model_path 路径改为你自己的

#### train

```python
python train_CME.py
```

#### predict

```
python predict_CME.py
```

### 效果

想打榜的，可以加**模型融合+对抗训练+远程监督**等等 trick！ 这个 67.942% 是纯模型的结果

![1643421416_1_.jpg](https://s2.loli.net/2022/01/29/QLFdefT46Aakm7N.png)
