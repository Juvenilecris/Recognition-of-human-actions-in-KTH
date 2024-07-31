"""加载制作好的数据集, 调整模型参数(遍历M,N值), 计算分类准确率, 并保存混淆矩阵到文件夹confusion_Matrix_six_hmm"""
import cv2
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans
from hmmlearn import hmm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os 
from itertools import chain
import pickle
#---------------------dt-----------------------
# 从.pkl文件加载数据
with open('data/data_dt_f.pkl', 'rb') as f:
     data = pickle.load(f)
n_features=data[399].shape[1]
#data=np.array(data)
with open('data/frames.pkl', 'rb') as f:
     frames = pickle.load(f)     
with open('data/labels_dt_f.pkl', 'rb') as f:
    labels = pickle.load(f)
labels=np.array(labels)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#------------------------划分各类数据，进行单独聚类-------------------
actions = ["walking", "boxing", "handclapping", "jogging", "running", "handwaving"]
data_walk=data[0:100]
data_box=data[100:200]
data_handclap=data[200:299]
data_jog=data[299:399]
data_run=data[399:499]
data_wave=data[499:599]

frames_walk=frames[0:100]
frames_box=frames[100:200]
frames_handclap=frames[200:299]
frames_jog=frames[299:399]
frames_run=frames[399:499]
frames_wave=frames[499:599]     

data_walk=np.vstack(data_walk)
data_box=np.vstack(data_box)
data_handclap=np.vstack(data_handclap)
data_jog=np.vstack(data_jog)
data_run=np.vstack(data_run)
data_wave=np.vstack(data_wave)
#-------------------------进行PCA降维---------------------------------
n_pca=260     #主成分个数
# 对每个数据点进行降维
pca = PCA(n_components=n_pca)
pca.fit(np.concatenate([data_walk,data_box,data_handclap,data_jog,data_run,data_wave]))
data_pca_walk = pca.transform(data_walk)
data_pca_box = pca.transform(data_box)
data_pca_handclap= pca.transform(data_handclap)
data_pca_jog = pca.transform(data_jog)
data_pca_run = pca.transform(data_run)
data_pca_wave = pca.transform(data_wave)

#-----------------------------------参数遍历------------------------------
for m in range(2,20,1):
    # ---------------执行 KMeans 聚类----------------
    use_pca=True
    num_clusters = m  
    
    kmeans_walk = KMeans(n_clusters=num_clusters, random_state=0,n_init=300)
    clusters_walk = kmeans_walk.fit_predict(data_pca_walk) if use_pca else  kmeans_walk.fit_predict(data_walk)
    
    kmeans_box = KMeans(n_clusters=num_clusters, random_state=0,n_init=300)
    clusters_box = kmeans_box.fit_predict(data_pca_box) if use_pca else  kmeans_walk.fit_predict(data_box)
    
    kmeans_handclap = KMeans(n_clusters=num_clusters, random_state=0,n_init=300)
    clusters_handclap = kmeans_handclap.fit_predict(data_pca_handclap) if use_pca else  kmeans_handclap.fit_predict(data_handclap)

    kmeans_jog = KMeans(n_clusters=num_clusters, random_state=0,n_init=300)
    clusters_jog = kmeans_jog.fit_predict(data_pca_jog) if use_pca else  kmeans_jog.fit_predict(data_jog)

    kmeans_run = KMeans(n_clusters=num_clusters, random_state=0,n_init=300)
    clusters_run = kmeans_run.fit_predict(data_pca_run) if use_pca else  kmeans_run.fit_predict(data_run)

    kmeans_wave = KMeans(n_clusters=num_clusters, random_state=0,n_init=300)
    clusters_wave = kmeans_wave.fit_predict(data_pca_wave) if use_pca else  kmeans_wave.fit_predict(data_wave)

    clusters_walk_ = []  # 用来存储分割后的序列列表
    start_idx = 0
    for frame in frames_walk:
        # 根据frames数组中的值，从data中分割出一个序列
        sequence = clusters_walk[start_idx:start_idx + frame]
        # 将该序列添加到sequences列表中
        clusters_walk_.append(sequence)
        # 更新起始索引
        start_idx += frame
    data_walk =clusters_walk_

    clusters_box_ = []  # 用来存储分割后的序列列表
    start_idx = 0
    for frame in frames_box:
        # 根据frames数组中的值，从data中分割出一个序列
        sequence = clusters_box[start_idx:start_idx + frame]
        # 将该序列添加到sequences列表中
        clusters_box_.append(sequence)
        # 更新起始索引
        start_idx += frame
    data_box =clusters_box_

    clusters_handclap_ = []  # 用来存储分割后的序列列表
    start_idx = 0
    for frame in frames_handclap:
        # 根据frames数组中的值，从data中分割出一个序列
        sequence = clusters_handclap[start_idx:start_idx + frame]
        # 将该序列添加到sequences列表中
        clusters_handclap_.append(sequence)
        # 更新起始索引
        start_idx += frame
    data_handclap =clusters_handclap_

    clusters_jog_ = []  # 用来存储分割后的序列列表
    start_idx = 0
    for frame in frames_jog:
        # 根据frames数组中的值，从data中分割出一个序列
        sequence = clusters_jog[start_idx:start_idx + frame]
        # 将该序列添加到sequences列表中
        clusters_jog_.append(sequence)
        # 更新起始索引
        start_idx += frame
    data_jog = clusters_jog_

    clusters_run_ = []  # 用来存储分割后的序列列表
    start_idx = 0
    for frame in frames_run:
        # 根据frames数组中的值，从data中分割出一个序列
        sequence = clusters_run[start_idx:start_idx + frame]
        # 将该序列添加到sequences列表中
        clusters_run_.append(sequence)
        # 更新起始索引
        start_idx += frame
    data_run= clusters_run_

    clusters_wave_ = []  # 用来存储分割后的序列列表
    start_idx = 0
    for frame in frames_wave:
        # 根据frames数组中的值，从data中分割出一个序列
        sequence = clusters_wave[start_idx:start_idx + frame]
        # 将该序列添加到sequences列表中
        clusters_wave_.append(sequence)
        # 更新起始索引
        start_idx += frame
    data_wave = clusters_wave_

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    # ----------------划分训练集和测试集-----------------
    split=0.5
    X_train_walk=data_walk[::2]
    X_test_walk=data_walk[1::2]

    X_train_box=data_box[::2]
    X_test_box=data_box[1::2]

    X_train_handclap=data_handclap[::2]
    X_test_handclap=data_handclap[1::2]

    X_train_jog=data_jog[::2]
    X_test_jog=data_jog[1::2]

    X_train_run=data_run[::2]
    X_test_run=data_run[1::2]

    X_train_wave=data_wave[::2]
    X_test_wave=data_wave[1::2]
    #print(X_train_handclap)
    #制作训练序列
    data_walk=np.concatenate([i for i in X_train_walk]).reshape(-1,1)
    data_box=np.concatenate([i for i in X_train_box]).reshape(-1,1)
    data_handclap=np.concatenate([i for i in X_train_handclap]).reshape(-1,1)
    data_jog=np.concatenate([i for i in X_train_jog]).reshape(-1,1)
    data_run=np.concatenate([i for i in X_train_run]).reshape(-1,1)
    data_wave=np.concatenate([i for i in X_train_wave]).reshape(-1,1)
    #序列长度
    x_len_walk=frames_walk[::2]
    x_len_box=frames_box[::2]
    x_len_handclap=frames_handclap[::2]
    x_len_jog=frames_jog[::2]
    x_len_run=frames_run[::2]
    x_len_wave=frames_wave[::2]
    
    x_len_walk_=frames_walk[1::2]
    x_len_box_=frames_box[1::2]
    x_len_handclap_=frames_handclap[1::2]
    x_len_jog_=frames_jog[1::2]
    x_len_run_=frames_run[1::2]
    x_len_wave_=frames_wave[1::2]

    labels_walk=labels[0:100]
    labels_box=labels[100:200]
    labels_handclap=labels[200:299]
    labels_jog=labels[299:399]
    labels_run=labels[399:499]
    labels_wave=labels[499:599]


    train_labels_walk=labels_walk[::2]
    train_labels_box=labels_box[::2]
    train_labels_handclap=labels_handclap[::2]
    train_labels_jog=labels_jog[::2]
    train_labels_run=labels_run[::2]
    train_labels_wave=labels_wave[::2]
    # X_train=X_train_walk+X_train_box+X_train_handclap+X_train_jog+X_train_run+X_train_wave
    # y_train=train_labels_walk+train_labels_box+train_labels_handclap+train_labels_jog+train_labels_run+train_labels_wave
    test_labels_walk=labels_walk[1::2]
    test_labels_box=labels_box[1::2]
    test_labels_handclap=labels_handclap[1::2]
    test_labels_jog=labels_jog[1::2]
    test_labels_run=labels_run[1::2]
    test_labels_wave=labels_wave[1::2]
    # X_test=X_test_walk+X_test_box+X_test_handclap+X_test_jog+X_test_run+X_test_wave
    # y_test=test_labels_walk+test_labels_box+test_labels_handclap+test_labels_jog+test_labels_run+test_labels_wave

    for h in range(2,20,1):
        from hmmlearn import hmm
        #-----------------训练HMM模型--------------------
        n_hmm=h#隐藏节点个数
        n_iter=300              #迭代最大个数   调大点 >200
        algorithm='map'     #解码算法   最大后验概率解码       'viterbi'维特比解码算法
        tol=0.001         #相邻两次迭代之间对数似然变化 迭代终止阈值
        init_params='sc' #stemc  stec-56% sec-57% sc-58% c-51%
        n_features=220 
        assert n_features>=num_clusters                
        model_walk = hmm.CategoricalHMM(n_components=n_hmm, n_iter=n_iter,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features)
        model_box = hmm.CategoricalHMM(n_components=n_hmm,  n_iter=n_iter,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features)
        model_handclap= hmm.CategoricalHMM(n_components=n_hmm, n_iter=n_iter,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features)
        model_jog = hmm.CategoricalHMM(n_components=n_hmm,  n_iter=n_iter,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features)
        model_run = hmm.CategoricalHMM(n_components=n_hmm, n_iter=n_iter,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features)
        model_handwave = hmm.CategoricalHMM(n_components=n_hmm,  n_iter=n_iter,algorithm=algorithm,verbose=False,tol=tol,init_params=init_params,n_features=n_features)

        model_walk.fit(data_walk,x_len_walk)
        model_box.fit(data_box,x_len_box)
        model_handclap.fit(data_handclap,x_len_handclap)
        model_jog.fit(data_jog,x_len_jog)
        model_run.fit(data_run,x_len_run)
        model_handwave.fit(data_wave,x_len_wave)
        #------------------------评估准确率-----------------------------
        correct_count = 0
        total_count = 300
        for i, x in enumerate(chain(X_train_walk,X_train_box,X_train_handclap,X_test_jog,X_test_run,X_test_wave)):
            # 计算两个类别模型预测的概率
            score=[model_walk.score(x.reshape(-1, 1)),model_box.score(x.reshape(-1, 1)),model_handclap.score(x.reshape(-1, 1)),
                model_jog.score(x.reshape(-1, 1)),model_run.score(x.reshape(-1, 1)),model_handwave.score(x.reshape(-1, 1))]
            
            predicted_label=score.index(max(score))
            if predicted_label == list(chain(train_labels_walk,train_labels_box,train_labels_box,train_labels_jog,train_labels_run,train_labels_wave))[i]:
                correct_count += 1

        # 计算分类器的准确率
        train_accuracy = correct_count / total_count
        print(f" M_{num_clusters}_N_{n_hmm}分类器在训练集的准确率：", train_accuracy)
    
        correct_count = 0
        total_count = 299
        predict_labels_test=[]
        for i, x in enumerate(chain(X_test_walk,X_test_box,X_test_handclap,X_test_jog,X_test_run,X_test_wave)):
            # 计算两个类别模型预测的概率
            score=[model_walk.score(x.reshape(-1, 1)),model_box.score(x.reshape(-1, 1)),model_handclap.score(x.reshape(-1, 1)),
                model_jog.score(x.reshape(-1, 1)),model_run.score(x.reshape(-1, 1)),model_handwave.score(x.reshape(-1, 1))]
            #print(score)           #对数似然值
            predicted_label=score.index(max(score))
            #print(predicted_label)
            predict_labels_test.append(predicted_label)
            if predicted_label == list(chain(test_labels_walk,test_labels_box,test_labels_handclap,test_labels_jog,test_labels_run,test_labels_wave))[i]:
                correct_count += 1

        # 计算分类器的准确率
        test_accuracy = correct_count / total_count
        print("分类器在测试集的准确率：", test_accuracy)
        
    
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        #----------------------------做出混淆矩阵-----------------------------
        conf_matrix = confusion_matrix(list(chain(test_labels_walk,test_labels_box,test_labels_handclap,test_labels_jog,test_labels_run,test_labels_wave)), predict_labels_test)

        plt.figure(figsize=(16, 12))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['walk', 'box', 'handclap', 'jog', 'run', 'handwave'], yticklabels=['walk', 'box', 'handclap', 'jog', 'run', 'handwave'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix in M={num_clusters},N={n_hmm}')
        plt.savefig(f'confusion_Matrix_six_hmm/M_{num_clusters}_N_{n_hmm}_{test_accuracy}.png')
