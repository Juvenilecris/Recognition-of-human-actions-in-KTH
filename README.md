# 基于隐马尔可夫模型的KTH动作视频识别
## 任务要求
    本项目基于隐马尔可夫模型实现视频中人体行为的识别任务，完成了基础题以及拔高题的所有内容。

    1.	下载KTH数据集文件，并从数据集的视频中提取特征；奇数索引的视频数据作为训练集，偶数索引的视频数据作为测试集。
    2.	在提取特征后的训练数据的基础上，实现一个空间聚类算法（例如 K-Means，任何聚类算法均可）。
    3.	为每类动作训练单独的隐马尔可夫模型（每个元音单独聚类，然后分别学习 HMM，输入是与每个2D点关联的聚类类别号）。对于每个测试数据，针对每个HMM计算其对数似然，即log P(O|M)，并获取给出最高对数似然的HMM类别，即对测试数据进行分类 判别。给出混淆矩阵并描述你的发现。
    4.	改变2中的聚类数量（变量M）和3中隐藏节点的数量（变量N）并计算分类准确率。给出不同M和N取值下的混淆矩阵，并描述你的发现。
    5.	实现动态时间规整算法（Dynamic Time Warping），并重复1-4。
    6.	学习一个HMM模型，该HMM可以使用维特比解码(Viterbi Decoding)来执行分类（注意：本题目不是让你生成多个分类器并进行分类判别，而是生成一个HMM模型，可以执行多个类别判别）。将分类准确率与题目 3 和4的结果进行比较，并描述你的发现。

## 实验结果
    1、实验1中，基于6个HMM模型的对数似然分类模型在测试集的分类准确率高达70%

    2、实验2中，基于DTW算法的分类模型在测试集的分类准确率高达57%
    
    3、实验3中，基于1个HMM模型的解码结果分类模型在测试集的分类准确率高达69%


## 代码参考说明

    1、hmm模型库：https://hmmlearn.readthedocs.io/en/latest/index.html
    2、基于DTW度量标准的KNN代码部分参考：https://blog.csdn.net/kuabiku/article/details/132732019，其中fastdtw计算库：https://pypi.org/project/fastdtw/
    3、特征提取部分代码参考Chatgpt提供的openCV方法以及opencv使用文档：https://apachecn.github.io/opencv-doc-zh/#/docs/4.0.0/5.4-tutorial_py_sift_intro
    3、轮廓系数计算和轮廓图绘制代码参考：https://blog.csdn.net/weixin_45275599/article/details/133705921
    4、SIFT特征可视化代码参考：https://blog.csdn.net/weixin_42795788/article/details/124244189