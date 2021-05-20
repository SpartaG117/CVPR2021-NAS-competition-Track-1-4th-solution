# CVPR2021-NAS-competition-Track-1-4th-solution

## 项目描述
首先采用权值共享的方式进行采样训练，然后将每层的权值按输出独立成operator的方式，对于相同输出不同的输入共享同一个operator权值。
采样采用uniform sample的方式，，每采样32个subnet更新一次参数。权值共享训练的stage初始学习率0.1， cosine scheduler, 300epoch, operator训练stage采用固定学习率0.001, 1000epoch.


## 项目结构
```
-|data
-|checkpoints
-README.MD
-merge.py
-gen_numpy_data.py
-model_op.py
-model.py
-submit_sh.py
-submit_single.py
-test.py
-train.py
-utils.py
```

## 使用方式 1

准备numpy数据集加速evaluation
```
python gen_numpy_data.py
```

训练 weight sharing style模型， lcm为每一个batch采样次数
```
python train.py --model_type 0 --output_dir ws --lcm 32 --epochs 300
```

可选：评测weight sharing style模型, bnbatch大小影响bn重标定速度
```
python test.py --model_type 0 --output_dir ws --bnbatch 20 --model_name epoch_299.pth
```

选取最好或最终ws模型，为operator style模型初始化进行finetune, pretrain为选择的初始化模型checkpoint
```
python train.py --model_type 1 --output_dir op --lcm 32 --epochs 1000 --fixlr --learning_rate 0.001 --pretrain ./checkpoints/ws/weights.pth
```

可选：评测最终模型, bnbatch大小影响bn重标定速度， model_name为checkpoint名称

```
python test.py --model_type 1 --output_dir op --bnbatch 20 --model_name epoch_299.pth
```

生成sh脚本进行50000个子模型进行evaluation, num_gpu为使用的gpu数量，n_p为开启的进程数
```
python submit_sh.py --num_gpu 4 --n_p 16 --output_dir op --model_name epoch_999.pth
```

执行sh生成脚本
```
bash ./checkpoints/op/submission/run_0_50000.sh
```

merge子json文件, 得到submission_final.json
```
python merge.py ./checkpoints/op/submission
```

## 使用方式 2

执行sh生成脚本

```
bash ./train_gen.sh
```

## 使用方式 3

线上最终提交版本权值为目录下weigths.pth, 放置于目录./checkpoints/op,可用如下方式进行生成

```
bash gen.sh
```