# cats_vs_dogs

数据集：kaggle 猫狗大战
训练集：https://www.kaggle.com/sbgyshds/cat-vs-dogs-with-small-datasets train_after_crop.zip  
验证集：https://www.kaggle.com/sbgyshds/cat-vs-dogs-with-small-datasets validation_with_group.zip  
模型下载：https://www.kaggle.com/sbgyshds/cat-vs-dogs-with-small-datasets models.zip  

## 模型说明
k_1_dnn_250_250.h5 dnn模型，img_size为250*250，5折验证  
k_2_dnn_250_250.h5 dnn模型，img_size为250*250，5折验证  
k_3_dnn_250_250.h5 dnn模型，img_size为250*250，5折验证  
k_4_dnn_250_250.h5 dnn模型，img_size为250*250，5折验证  
k_5_dnn_250_250.h5 dnn模型，img_size为250*250，5折验证  

## 模型下载
链接: https://pan.baidu.com/s/1hIJVAX082FM6WaVI25paUg 提取码: fd76

## 实验结果解读

### 实验目标
阅读keras文档中的小数据集深度学习一文，了解目前能到达的精度，  
发现在借助迁移学习的前提下，能轻松到达百分之94的准确率。  
但是迁移学习的数据集是imagenet，含有猫狗，因此提出问题，不借助迁移学习能将精度提高到多少

### 实验准备
数据集选择了猫狗各1000张图片，模型选择keras文档中的三层卷积网络模型、alexnet与17层的cnn网络。
超参数方面，经过简单试验初步选择，决定采用dnn与(250,250)的图像大小。  
从试验结果来看，越深的网络适用的图像大小越大，数据增广对小数据集的提升极大

### 数据清洗
用作者本人开发的bbox工具将图片中的猫狗裁剪出，并剔除一部分有问题的训练数据。  
从原始数据来看，猫的图片中影响结果的噪声较多（比如人的手），狗相对好一些

###第一次训练
采用5折验证，每一次的训练集准确率能到达98%，验证集准确率为88%-90%，从验证集的结果来看，  
模型提取的姿态方面的特征较多，对光照与猫狗的部位细节的特征提取的较为不好.  
若将特征向量用svm进行分类，准确率大概为78%.  

### 进一步优化
细粒度图像分类模型(MASK_CNN,双线性CNN)

### 第二次训练
将前一步骤中已经训练好的网络通过bilinear方式进行连接。
此时再将整个网络训练几个epoch（会存在过拟合），验证集上的loss能下降到0.32  

由于一次操作失误，误利用未经bbox裁剪的数据进行训练，发现能够将loss突破至0.3以下，  
经过思考与实验，将训练集设置为未经bbox的数值，数据增广的幅度减小一半（各项系数减半），  
反复调整bilinear cnn训练时的learning rate，得到验证集0.26左右的loss，提交至kaggle，结果为0.26217。（单折模型）
验证集的准确度则基本在90%以上

五折模型各自的loss分别为0.26217, 0.23804, 0.29331, 0.26660, 0.30656
将结果限定在[0.005,0.995]之后，结果大概会提升千分点的样子，这不是我们现在的瓶颈所在。  
可以猜测模型预测的正确率尚可，但是对于预测结果不够肯定。

### 模型融合
融合之后的模型的提交loss为：  
平均融合0.21060  
平均融合(将结果限定在[0.005,0.995]间)0.20927  
平均融合(结果为离散值0,1)2.70787  
平均融合(结果为离散值0.005,0.995)0.42  
投票融合2.99246  
根据结果的loss反推正确率可得在测试集上的结果大概为92%

## 实验结论
优化器无脑adam，最后一层激活函数无脑sigmoid（二分类），batch_size设成2的幂。  
数据增广和bbox效果拔群，图像大小和网络的深度正相关，