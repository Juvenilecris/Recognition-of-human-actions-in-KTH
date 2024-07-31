#基于CV库分别提取视频各帧（以1秒为时间步）的dense_trajectories特征，HOG特征，SIFT特征，之后将对三种特征下的模型进行评估
#参考ChatGpt询问到的CV库中提取这些特征的方法
import cv2
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans
from hmmlearn import hmm
import os 

"""从视频中按帧/s 提取Dense Trajectories特征"""
def extract_features_dt_f(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    dense_sampling_interval = frame_rate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, frame_rate * 100)  
    dense_trajectories = []
    prev_gray = None
    frame_number = 0
    hsv = np.zeros((160, 120, 3))
    while frame_number < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (120, 160))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            dense_trajectories.append(hsv.flatten())
        
        prev_gray = gray
        frame_number += dense_sampling_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    cap.release()
    return np.array(dense_trajectories),np.array(dense_trajectories).shape[0]

"""从视频中按帧/s 提取HOG特征"""
def extract_features_hog(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, frame_rate * 100)  
    samples_per_second = frame_count / total_frames
    hog = cv2.HOGDescriptor((72,96),(12,12),(12,12),(6,6),9)
    features = []
    print(total_frames)
    for i in range(0,total_frames, frame_rate):
        frame_index = int(i * samples_per_second)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (120, 160))
        resized_frame = np.array(resized_frame * 255, dtype=np.uint8)  
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        hog_features = hog.compute(gray_frame,(36,36),padding=(12,12))
        features.append(hog_features.flatten())
    cap.release()
    return features, len(features)

"""从视频中按帧/s 提取SIFT特征"""
def extract_features_sift(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frame_count, frame_rate * 100)  
    samples_per_second = frame_count / total_frames
    features = []
    sift = cv2.SIFT_create()
    for i in range(0,total_frames, frame_rate):
        frame_index = int(i * samples_per_second)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
             break
        
        resized_frame = cv2.resize(frame, (120, 160))
        resized_frame = np.array(resized_frame * 255, dtype=np.uint8)  
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
      
        if descriptors is not None:
             features.append(descriptors.flatten())
        else:
            features.append(np.array([0,0]).astype(np.float32)) 
    cap.release()
    return np.array(features), np.array(features).shape[0]

if __name__ == '__main__':
    # 从数据集中提取数据
    data = []
    labels = []
    frames=[]
    data_dir = "data"
    actions = ["walking", "boxing", "handclapping", "jogging", "running", "handwaving"]
    for action_label, action in enumerate(actions):
        action_dir = os.path.join(data_dir, action)
        for filename in os.listdir(action_dir):
            video_path = os.path.join(action_dir, filename)
            # 从视频中提取特征
            video_features,frame = extract_features_dt_f(video_path)       
            data.append(video_features)
            labels.append(action_label)
            frames.append(frame)
    # 检查数据长度，存储数据集至文件
    import pickle
    print(len(data))
    with open('data/data_dt_f.pkl', 'wb') as f:
        pickle.dump(data, f)
    # 将数据存储为.pkl文件
    with open('data/labels_dt_f.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('data/frames.pkl', 'wb') as f:
        pickle.dump(frames, f)