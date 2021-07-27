# Ysneaker—球鞋智能辅助鉴定平台

电子商务的发展虽然极大地方便了人们消费，但是非面对面的交易使得假货问题频出，调查发现商品真伪鉴定困难是市场中假货泛滥的主要因素。传统的真伪鉴定主要依靠专业机构人工进行，但此类方法存在着1.大批量鉴定费用昂贵，2.消费市场和专家数量不匹配，3.鉴定程序复杂，周期较长等问题。面对上述问题，并结合近些年球鞋交易市场的火爆及高仿球鞋的鉴定难度，本团队从球鞋领域入手，研发了Ysneaker—球鞋智能辅助鉴定平台，向用户提供了球鞋真伪鉴定、球鞋检索和球鞋风格迁移三种功能。此存储库为Ysneaker平台球鞋鉴定、检索多视图多任务模型的Paddle代码实现，完整功能详见[Ysneaker网站](http://www.ysneaker.com)，欢迎访问！也可以通过[AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2215997 )进行简单的了解。

## 依赖

- python = 3.6
- paddle = 2.1
- visualdl = 2.2.0

## 文件目录
<pre>
<font color="#729FCF"><b>.</b></font>
├── <font color="#729FCF"><b>data</b></font> 
│   ├── <font color="#729FCF"><b>images</b></font>           # 球鞋数据集
│   └── xxxx.json        # 图片-标签文件
├── <font color="#729FCF"><b>model</b></font>                # 预训练模型
├── <font color="#729FCF"><b>vdl</b></font>                  # Loss可视化
├── dataset.py           # 数据处理和加载
├── model.py             # 模型结构
├── train.py             # 训练
├── eval.py              # 测试
└── README.md
</pre>

## 模型训练
### 准备数据
#### 多视图球鞋鉴定
将多视图球鞋数据集放入`./data/images/`文件夹，构造存储图片-标签的`json`文件，并将该文件放入`./data/`文件夹下，
json文件结构如下：
```bash
{'img': [['appearance.png', 'box_logo.png', 'insole.png', 'midsole.png', 'stamp.png', 'tag.png'], [...], [...], ...],'label': [1, 0, 1, ...]]}
```
（如./data/identify_train.json所示）

#### 球鞋检索
球鞋检索数据来自多视图数据中的`appearance.png`，因此只需要构造图片-标签`json`文件，并将该文件放入`./data/`文件夹下即可，
json文件结构如下：
```bash
{'appearance.png': 5, 'appearance_1.png': 2, 'appearance_2.png': 10，......}
```
（如./data/retrieval_train.json所示）

### 训练
单卡 
```bash
CUDA_VISIBLE_DEVICES = 0 python train.py --dataset_path ./data/images/ --identify_path_train ./data/xxxx.json --retrieval_path_train ./data/xxxx.json --identify_path_val ./data/xxxx.json --retrieval_path_val  ./data/xxxx.json --batch_size xx --epoch xx
```
多卡 
```bash
python -m paddle.distributed.launch train.py --dataset_path ./data/images/ --identify_path_train ./data/xxxx.json --retrieval_path_train ./data/xxxx.json --identify_path_val ./data/xxxx.json --retrieval_path_val  ./data/xxxx.json --batch_size xx --epoch xx
```
Loss可视化 
```bash
visualdl --logdir ./vdl/ --host 0.0.0.0 --port 8080
```
我们还提供了完整的预训练模型，点击[此处](https://pan.baidu.com/s/1mfKTbYRrjGDOIS2Wat9Vtw )下载，提取码为3gmg。

### 测试
```bash
CUDA_VISIBLE_DEVICES = 0 python eval.py --dataset_path ./data/images/ --identify_path_test ./data/xxxx.json --retrieval_path_test ./data/xxxx.json --batch_size xx
```

### 性能
球鞋鉴定
```python
{'F1 Score Macro': 0.873, 'Precision Score Macro': 0.871, 'Recall Score Macro': 0.874}
```
球鞋检索
```python
{'F1 Score': 0.800, 'Precision Score': 0.800, 'Recall Score': 0.849}
```

# Demo
## 训练
```bash
CUDA_VISIBLE_DEVICES = 0 python train.py  --dataset_path ./data/images/ --identify_path_train ./data/identify_train.json --retrieval_path_train ./data/retrieval_train.json  --identify_path_val ./data/identify_val.json --retrieval_path_val  ./data/retrieval_val.json --batch_size 1 --epoch 10
```

## 测试
```bash
CUDA_VISIBLE_DEVICES = 0 python eval.py --dataset_path ./data/images/ --identify_path_test ./data/identify_test.json --retrieval_path_test  ./data/retrieval_test.json --batch_size 1
```

# 致谢
感谢百度PaddlePaddle深度学习框架提供的支持。