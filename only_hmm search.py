"""加载制作好的数据集, 调整模型参数(遍历M,N值), 计算分类准确率, 并保存混淆矩阵到文件夹confusion_Matrix_only_hmm"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from hmmlearn import hmm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import numpy as np
from fastdtw import fastdtw
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#-----------------------定义DTW度量标准的KNN分类器------------------
def dtw_distance(x1, x2):
    # 计算动态时间规整距离
    alignment = fastdtw(x1, x2)[1]
    dtw_dist = 0
    for i, j in alignment:
        dtw_dist += abs(x1[i] - x2[j])
    return dtw_dist
class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    def _predict(self, x):
        # 计算样本与所有训练数据的DTW距离
        distances = [dtw_distance(x, x_train) for x_train in self.X_train]
        # 根据距离排序，找到K个最近邻的样本
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 通过投票机制确定新样本的类别
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
#---------------------加载数据集-----------------------
with open('Data/data_dt_f.pkl', 'rb') as f:
     data = pickle.load(f)
n_features=data[399].shape[1]
data=np.array(data)
with open('Data/labels_dt.pkl', 'rb') as f:
    labels = pickle.load(f)
with open('Data/frames.pkl', 'rb') as f:
     frames = pickle.load(f)

actions = ["walking", "boxing", "handclapping", "jogging", "running", "handwaving"]
data=np.vstack(data)

labels_ = []  # 用来存储分割后的序列列表
for i,label in enumerate(labels):
    for j in range(frames[i]):
        labels_.append(label)
labels_ = np.array(labels_)
#--------------主成分降维-----------
n_pca=260     #主成分个数
# 对每个数据点进行降维
pca = PCA(n_components=n_pca)
data_pca=pca.fit_transform(data)  
n_lda=5
#---------------LDA降维--------------
lda =  LinearDiscriminantAnalysis(n_components=n_lda)
lda.fit(data_pca,labels_)
data_lda = lda.transform(data_pca)
   
labels=np.array(labels)
frames_train=frames[::2]
frames_test=frames[1::2]
labels_train=labels[::2]
labels_test=labels[1::2]
#-------------------------遍历参数组合-------------------------
for m in range(10,200,4):
    try:
        # ---------------执行 KMeans聚类 --------------
        num_clusters_=m
        kmeans = KMeans(n_clusters=num_clusters_, random_state=0,n_init=500)
        clusters=kmeans.fit_predict(data_lda)
        clusters_walk_ =[]  # 用来存储分割后的序列列表
        start_idx = 0
        clusters_ = []  # 用来存储分割后的序列列表
        #--------------------划分序列--------------
        for frame in frames:
            # 根据frames数组中的值，从data中分割出一个序列
            sequence = clusters[start_idx:start_idx + frame]
            # 将该序列添加到sequences列表中
            clusters_.append(sequence)
            # 更新起始索引
            start_idx += frame
        data_all = np.array(clusters_,dtype=object)
        data_train=data_all[::2]
        data_test=data_all[1::2]
        data_hmm=np.concatenate([i for i in data_train]).reshape(-1,1)
        for h in range(8,int(m*2),3):
            try:
                #---------------------训练HMM模型-------------------
                n_hmm_=h
                n_iter_=350
                algorithm='map'
                tol=0.001
                init_params='sc'
                n_features_=260
                model_hmm = hmm.CategoricalHMM(n_components=n_hmm_, n_iter=n_iter_,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features_)
                model_hmm.fit(data_hmm,frames_train)
                
                #----------------------对观测序列进行解码----------------
                train_decodered=[]
                for i, x in enumerate(data_train):    
                    train_decodered.append(model_hmm.decode(x.reshape(-1,1),frames_train[i],algorithm='viterbi')[1])
                test_decodered=[]
                for i, x in enumerate(data_test):    
                    test_decodered.append(model_hmm.decode(x.reshape(-1,1),frames_test[i],algorithm='viterbi')[1])
                # ----------------基于DTW的KNN对解码序列进行分类-------------
                k=40
                knn = KNN(k=k)
                knn.fit(train_decodered, labels_train)
                #--------------评估准确率-----------------
                predictions = knn.predict(test_decodered)
                predictions_train = knn.predict(train_decodered)
                count=0
                total_count=len(predictions_train)
                for i,prediction in enumerate(predictions_train):
                    if prediction==labels_train[i]:
                        count+=1
                train_accuracy=count/total_count
                print(f"在训练集上的Accuracy: {train_accuracy}")
                count=0
                total_count=len(predictions)
                for i,prediction in enumerate(predictions):
                    if prediction==labels_test[i]:
                        count+=1
                test_accuracy=count/total_count
                print(f"在测试集上的Accuracy: {test_accuracy}")
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.metrics import confusion_matrix
                conf_matrix = confusion_matrix(labels_test,predictions)

                # 绘制热力图
                plt.figure(figsize=(16, 12))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['walk', 'box', 'handclap', 'jog', 'run', 'handwave'], yticklabels=['walk', 'box', 'handclap', 'jog', 'run', 'handwave'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix in M={num_clusters_},N={n_hmm_}')
                plt.savefig(f'E:/机器学习实验/Project/Recognition_of_human_actions/confusion_Matrix_only_hmm/M={num_clusters_},N={n_hmm_},{test_accuracy}.png')
            except:
                pass
    except:
        pass        