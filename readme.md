表征学习行人重识别系统baseline

包含文件：

data_loader.py    构建适合re-ID任务的数据集ImageDataset类

models                 包含神经网络模型

eval_metrics.py  评价程序

data_manager.py 用于最初读取数据库的信息

train_class.py        代码主程序，包括main，train，test

util.py  包含各种小工具AverageMeter，readImage等小函数

losses 使用crosstropy交叉熵分类损失

optimizer 和transforms利用torch自带的实现。



训练代码
CUDA_VISIBLE_DEVICES=2 python /data3/yiyuchen/try/train_class.py \
--root       /data3/yiyuchen/dl_datasets \
--dataset  market1501 \
--workers 4 \
--arch resnet50 \
--save-dir /data3/yiyuchen/try/log \
--gpu-devices 2

--max-epoch 1 \
--start-eval 80 \
--eval-step 10

从存档点处重新运行
CUDA_VISIBLE_DEVICES=2 python /data3/yiyuchen/try/train_class.py \
--root       /data3/yiyuchen/dl_datasets \
--resume  /data3/yiyuchen/try/log/best_model.pth.tar \
--dataset  market1501 \
--workers 4 \
--arch resnet50 \
--save-dir /data3/yiyuchen/try/log  \
--gpu-devices 2 \ 

--start-eval 80  \
--eval-step 10 \