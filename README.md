### Requirements
+ python 3.8
+ pickle 4.0
+ pandas 1.5.3
+ numpy 1.26.4
+ tensorflow >=2.7.0
+ keras >=2.7.0
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE could be reconfigured by user

### Usage
| 文件              | 用途                                                         |
| :---------------- | :----------------------------------------------------------- |
| checkVersion.py   | 用于检测环境是否一致，避免包依赖产生的错误。                 |
| loadData.py       | 将ASTENS文件载入数据集，所有源代码文件需使用astensEncoder将源代码转换为抽象语法树的编码序列，文件名以“labelID@filename.c”的形式组织，其中“labelID”表示分类标签，以正整数区分不同类别([1...N])。 |
| splitData.py      | 对数据集分类别按比例进行分割，“ratio='6:2:2'”表示训练数据、验证数据、测试数据的比例为60%、20%、20%，并在所有分类中保持同等比例。 |
| trainingModel.py  | 根据数据集训练模型，maxSeqLen设定ASTENS序列的最大长度，classNum设定类别数，hiddenUnits为BiLSTM层中隐节点数量，dropout为Dropout层比例，batchSize设定批大小，MaxEpoch设定最大训练轮次。 |
| prediction.py     | 使用预训练模型对现有数据进行代码功能性预测。                 |
| autoExperiment.sh | 自动进行多次模型训练以实现交叉验证，每次重新分割数据集，并重新训练模型。 |

#### 相关文献：

1. [ASTSDL: Predicting the Functionality of Incomplete Programming Code via AST-Sequence-based Deep Learning Model](http://engine.scichina.com/doi/10.1007/s11432-021-3665-1)
