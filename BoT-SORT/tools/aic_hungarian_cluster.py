'''
2023 aicity challenge

Using auto anchor generation.

input:single camera tracking results, tracklets from BoT-SORT and embedding
output:produce MCMT result at hungarian

'''
import pickle
import os
import collections
import argparse
import numpy as np
from loguru import logger
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('.')

def make_parser():
    parser = argparse.ArgumentParser("clustering for synthetic data")
    parser.add_argument("root_path", default="/home/hsiangwei/Desktop/AICITY2023/data", type=str)
    return parser

# 非极大值抑制（NMS），用于从一组重叠的边界框中选择最佳的边界框
def nms_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    # 如果boxes中的边界框坐标是整数类型，将它们转换为浮点数，因为接下来会进行除法运算
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    # 计算每个边界框的面积，并获取用于排序的索引。如果没有提供概率（probs），则默认按照边界框的右下角y坐标排序。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    # 如果提供了概率，则按照概率排序
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        # 从索引列表的末尾获取一个索引，将其加入pick列表
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        # 计算当前边界框与其它边界框的交集，包括最大的x、y起点和最小的x、y终点
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # 计算交集区域的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        # 计算重叠比例
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        # 删除所有重叠比例高于阈值overlapThresh的边界框索引
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked
    # 返回被选中的边界框及其索引
    return boxes[pick].astype("float"), pick

def get_people(scene, dataset, threshold):
    scenes = ['S003','S009','S014','S018','S021','S022']
    assert scene in scenes
    distance_thers = [13,19.5,16,13,16,16]
    # 通过scenes.index(scene)找到当前场景在列表中的索引
    seq_idx = scenes.index(scene)
    # 读取检测结果
    detections = np.genfromtxt(root_path+'/test_det/{}.txt'.format(scene), delimiter=',', dtype=str)
    # 加载嵌入向量
    embeddings = np.load(root_path+'/test_emb/{}.npy'.format(scene),allow_pickle = True)

    '''
    nms
    '''
    # 将嵌入向量转换为列表后又转换回numpy数组
    embeddings = embeddings.tolist()
    embeddings = np.array(embeddings)

    all_dets = None
    all_embs = None

    #  对于在特定阈值列表中的每一帧进行循环
    for frame in threshold[seq_idx][0]:

        # 获取当前帧的检测结果
        inds = detections[:,1] == str(frame-1)
        #  根据索引获取当前帧的检测结果和嵌入向量
        frame_detections = detections[inds]
        frame_embeddings = embeddings[inds]

        # 获取所有独特的摄像头编号
        cams = np.unique(detections[:,0])

        # 对每个摄像头的检测结果应用非极大值抑制（NMS），并根据阈值过滤结果，然后将结果添加到all_dets和all_embs中
        for cam in cams:
            inds = frame_detections[:,0]==cam
            cam_det = frame_detections[inds][:,1:].astype("float")
            cam_embedding = frame_embeddings[inds]
            cam_det,pick = nms_fast(cam_det,None,nms_thres)
            cam_embedding = cam_embedding[pick]
            if len(cam_det) == 0:continue
            inds = cam_det[:,6]>threshold[seq_idx][1]
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            if all_dets is None:
                all_dets = cam_det
                all_embs = cam_embedding
            else:
                all_dets = np.vstack((all_dets,cam_det))
                all_embs = np.vstack((all_embs,cam_embedding))

    #  使用层次聚类算法对所有嵌入向量进行聚类
    clustering = AgglomerativeClustering(distance_threshold=distance_thers[seq_idx],n_clusters=None).fit(all_embs)

    # 返回聚类结果中最大的标签编号加1，这代表了场景中的人数
    return max(clustering.labels_)+1

def get_anchor(scene, dataset, threshold, nms_thres):
    
    '''
    
    input: scene
    output: dictionary (keys: anchor's global id, values: a list of embeddings for that anchor)
    
    '''
    
    # 如果数据集是'test'，则定义了一个包含特定场景编号的列表scenes。如果不是'test'数据集，抛出一个错误，表明该数据集不被支持。
    if dataset == 'test':
        scenes = ['S003','S009','S014','S018','S021','S022']
    else:
        raise ValueError('{} not supported dataset!'.format(dataset))
    

    if scene in scenes:
        seq_idx = scenes.index(scene)
    else:
        raise ValueError('scene not in {} set!'.format(dataset))

    # 根据场景编号获取对应的场景。调用get_people函数，获取场景中的人数
    scene = scenes[seq_idx]
    k = get_people(scene,dataset,threshold)

    detections = np.genfromtxt(root_path+'/test_det/{}.txt'.format(scene), delimiter=',', dtype=str)
    embeddings = np.load(root_path+'/test_emb/{}.npy'.format(scene),allow_pickle = True)

    '''
    nms
    '''
    embeddings = embeddings.tolist()
    embeddings = np.array(embeddings)

    all_dets = None
    all_embs = None

    # 遍历特定帧的阈值设置。对每个帧中的每个摄像头应用非极大值抑制，并更新嵌入向量
    for frame in threshold[seq_idx][0]:

        inds = detections[:,1] == str(frame-1)
        frame_detections = detections[inds]
        frame_embeddings = embeddings[inds]

        cams = np.unique(detections[:,0])

        for cam in cams:
            inds = frame_detections[:,0]==cam
            cam_det = frame_detections[inds][:,1:].astype("float")
            cam_embedding = frame_embeddings[inds]
            cam_det,pick = nms_fast(cam_det,None,nms_thres)
            cam_embedding = cam_embedding[pick]
            if len(cam_det) == 0:continue
            inds = cam_det[:,6]>threshold[seq_idx][1]
            cam_det = cam_det[inds]
            cam_embedding = cam_embedding[inds]
            if all_dets is None:
                all_dets = cam_det
                all_embs = cam_embedding
            else:
                all_dets = np.vstack((all_dets,cam_det))
                all_embs = np.vstack((all_embs,cam_embedding))

            #cam_record += [cam]*len(cam_det)

    # 使用层次聚类对所有嵌入向量进行聚类
    clustering = AgglomerativeClustering(n_clusters=k).fit(all_embs)

    # print(clustering.labels_)

    # 创建一个默认字典anchors，用于存储每个锚点的嵌入向量
    anchors = collections.defaultdict(list)

    # 遍历每个锚点ID，将对应的嵌入向量添加到anchors字典中
    for global_id in range(k):
        for n in range(len(all_embs)):
            if global_id == clustering.labels_[n]:
                anchors[global_id].append(all_embs[n])
    
    return anchors

# 计算特征向量与多个锚点之间的距离
def get_box_dist(feat,anchors):
    '''
    input : feature, anchors                                                anc1  anc2  anc3 ...
    output : a list with distance between feature and anchor, e.g.    feat [dist1,dist2,dist3,...]
    '''
    box_dist = []

    # 遍历anchors字典中的每一个锚点
    for idx in anchors:
        dists = []
        # 遍历该锚点下的所有嵌入向量
        for anchor in anchors[idx]:
            #  将每个锚点向量标准化（除以其范数）
            anchor /= np.linalg.norm(anchor)
            # 计算特征向量feat与标准化后的锚点向量之间的余弦距离，并添加到dists列表中
            dists += [distance.cosine(feat,anchor)]
        
        #dist = min(dists) # or average..?
        # 计算到当前锚点的所有距离的平均值
        dist = sum(dists)/len(dists)
        
        box_dist.append(dist)    
            
    return box_dist
        

if __name__ == "__main__":
    
    args = make_parser().parse_args()
    root_path = args.root_path

    os.makedirs(os.path.join(root_path,'hungarian_cluster'),exist_ok=True)

    n = 15
    nms_thres = 1
    #nms_thres = 0.7
    #nms_thres = 0.3
    
    #dataset = 'validation'
    dataset = 'test'
    
    scene_path =  os.path.join(root_path,dataset)
    
    demo_synthetic_only = True
    
    if demo_synthetic_only:
        if dataset == 'test':
            scenes = ['S003','S009','S014','S018','S021','S022']
        else:
            scenes = ['S005','S008','S013','S017']

    # test
    threshold = [([1,5000,10000,15000],0.9), #OK
             ([1,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000],0.92), #OK
             ([1,2500,5000,7500,10000,15000],0.96), #OK
             ([1,5000,10000,15000],0.9), #OK
             ([1,2500,5000,7500,10000,12500,15000],0.95), #OK
             ([1,2500,5000,7500,10000],0.9)] #OK
    
    logger.info('clustering {} set'.format(dataset))
    logger.info('scenes list: {}'.format(scenes))
    logger.info('n = {}'.format(n))
    logger.info('nms_thres = {}'.format(nms_thres))
    logger.info('demo_synthetic_only = {}'.format(demo_synthetic_only))
    logger.info('threshold = {}'.format(threshold))
    
    for scene in scenes:
        
        anchors = get_anchor(scene,dataset,threshold,nms_thres)
        people = get_people(scene,dataset,threshold)
        logger.info('number of anchors {}'.format(len(anchors)))
        logger.info('number of people {}'.format(people))
        
        for anchor in anchors:
            logger.info('anchor {} : number of features {}'.format(anchor,len(anchors[anchor])))
        
        cams = os.listdir(os.path.join(scene_path,scene))
        cams = sorted([cam for cam in cams if not cam.endswith('png')])
        
        for cam in cams:
       
            logger.info('processing scene {} cam {}'.format(scene,cam))
            
            with open(root_path+'/tracklet/{}_{}.pkl'.format(scene,cam),'rb') as f:
                tracklets = pickle.load(f)
    
            mapper = collections.defaultdict(list)
            global_id_mapper = collections.defaultdict(list)
            
            for i,trk_id in enumerate(tracklets):
                if i%100==0:
                    logger.info('Progress: {}/{}'.format(i,len(tracklets)))
                trk = tracklets[trk_id]
                for feat in trk.features:
                    box_dist = get_box_dist(feat,anchors)
                    mapper[trk_id].append(box_dist)
                mapper[trk_id].append(box_dist) # extra length to prevent bug -- len(features) < len(tracklets) 
            sct = np.genfromtxt(root_path+'/SCT/{}_{}.txt'.format(scene,cam), delimiter=',', dtype=None)

            counter = collections.defaultdict(int)
            
            new_sct = root_path+'/hungarian_cluster/{}_{}.txt'.format(scene,cam)

            results = []
                
            cur_frame = -1
            cost_matrix = None

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in sct:

                if len(tracklets[trk_id].features) == 0: # too short trajectories, not sure if this can caused bug!
                    continue
                
                if frame_id != cur_frame and cost_matrix is None: # first frame
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                
                elif frame_id != cur_frame and cost_matrix is not None: # next frame => conduct hungarian, clear cost matrix
                    cost_matrix = np.array(cost_matrix)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for row,col in zip(row_ind,col_ind):
                        global_id_mapper[frame_trk_ids[row]].append(col)
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                cost_matrix.append(mapper[trk_id][counter[trk_id]])   # row is cost for this feature to all anchors
                frame_trk_ids.append(trk_id)

                counter[trk_id] += 1

            # for trk_id in global_id_mapper:
            #     print(trk_id,Counter(global_id_mapper[trk_id]).most_common())

            new_global_id_mapper = collections.defaultdict(list)

            for trk_id in global_id_mapper:
                ids = global_id_mapper[trk_id] # id list
                new_ids = []
                cur_ids = []
                for id in ids:
                    cur_ids.append(id)
                    if len(cur_ids) == n:
                        new_ids += [Counter(cur_ids).most_common(1)[0][0]]*n
                        cur_ids = []
                        
                if len(cur_ids)>0:
                    if len(new_ids)>0:
                        new_ids += [new_ids[-1]]*(len(cur_ids)+1)
                    else:
                        new_ids += [Counter(cur_ids).most_common(1)[0][0]]*(len(cur_ids)+1)
                new_global_id_mapper[trk_id] = new_ids
                new_global_id_mapper[trk_id] += [new_ids[-1]]*n # some extra length to prevent bug
                

            # Write Result

            conflict = 0
            counter2 = collections.defaultdict(int)

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in sct:
                
                if len(new_global_id_mapper[trk_id]) == 0:
                    continue
                
                if counter2[trk_id]>=len(new_global_id_mapper[trk_id]):
                    global_id = new_global_id_mapper[trk_id][-1]+1
                else:
                    global_id = new_global_id_mapper[trk_id][counter2[trk_id]]+1

                if frame_id != cur_frame:
                    global_ids = set()
                    cur_frame = frame_id
                if global_id in global_ids:
                    conflict+=1
                else:
                    global_ids.add(global_id)
                
                results.append(
                        f"{frame_id},{global_id},{x},{y},{w},{h},{score},-1,-1,-1\n"
                    )
                
                counter2[trk_id] += 1
            
            logger.info('conflict:{}'.format(conflict))
            
            with open(new_sct, 'w') as f:
                f.writelines(results)
