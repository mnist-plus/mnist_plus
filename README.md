# MNIST_plus

哈工大2018微软暑期课程项目

#### Requirements

- tensorflow >= 1.2

- opencv

#### 如何训练模型

```bash
python3 train_model.py
```

##### 查看训练结果

```
python3 plot.py logs --value minibatch_loss
python3 plot.py logs --value minibatch_acc
```

#### 如何使用已训练模型

在网盘下载参数文件，放在 checkpoints 文件中

链接: https://pan.baidu.com/s/1YvJmPT7-THFlWkWaD9MgjA 密码: rrij

并执行

```bash
python3 run_webapp.py
```

进入web端使用