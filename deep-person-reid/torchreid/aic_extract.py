'''
extract ReID features from testing data.
'''
import os
import os.path as osp
import argparse
import numpy as np
import torch
import time
import torchvision.transforms as T
from PIL import Image
from argparse import ArgumentParser
import sys
from utils import FeatureExtractor
import torchreid

def make_parser():
    parser = argparse.ArgumentParser("reid")
    parser.add_argument("root_path", type=str, default=None)
    #parser.add_argument('--reid_model_ckpt', help='ReID model path')
    parser.add_argument('--scene', default = None, help='scene')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = make_parser().parse_args()
    data_root = args.root_path

    test_list = ['S003','S009','S014','S018','S021','S022']

    dataset = 'test'

    data_root = args.root_path

    sys.path.append(data_root+'/deep-person-reid')

    img_dir = os.path.join(data_root,'data/{}'.format(dataset))
    det_dir = os.path.join(data_root,'data/{}_det'.format(dataset))
    out_dir = os.path.join(data_root,'data/{}_emb'.format(dataset))
    
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    reid_model_ckpt = osp.join(data_root,'deep_person_reid','checkpoints','synthetic_reid_model_60_epoch.pth')

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=reid_model_ckpt,
        device='cuda'
    )   
    
    scenes = test_list
    
    print(scenes)
    
    for scene in scenes:

        # 如果指定了特定场景并且当前场景与之不匹配，则跳过当前迭代
        if args.scene and scene!=args.scene:continue
            
        out_path = os.path.join(out_dir,scene+'.npy')
        det_path = os.path.join(det_dir,scene+'.txt')
        dets = np.genfromtxt(det_path,dtype=str,delimiter=',')
        cur_frame = 0
        # 初始化一个长度等于检测结果数量的NumPy数组
        emb = np.array([None]*len(dets))
        start = time.time()
        print('processing scene {} with {} detections'.format(scene,len(dets)))
        for idx,(cam,frame,_,x1,y1,x2,y2,_) in enumerate(dets):
            x1,y1,x2,y2 = map(float,[x1,y1,x2,y2])
            # 每处理1000个检测，打印处理时间和进度
            if idx%1000 == 0:
                end = time.time()
                print('processing time :',end-start)
                start = time.time()
                print('process {}/{}'.format(idx,len(dets)))

            # 检查当前帧号是否与检测数据中的帧号不同,如果帧号不同，则更新当前帧号
            if cur_frame != int(frame):
                cur_frame = int(frame)
                # 创建图像文件的路径,frame.zfill(5)确保帧号为5位数字（如果需要的话，前面补零）
                img_path = os.path.join(img_dir,scene,cam,'frame',frame.zfill(5)+'.jpg')
                img = Image.open(img_path)

            # 根据边界框坐标裁剪图像
            img_crop = img.crop((x1,y1,x2,y2))
            # 将裁剪后的图像转换为RGB，应用预定义的图像转换val_transforms，并增加一个新的维度
            img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
            feature = extractor(img_crop).cpu().detach().numpy()[0]
    
            # feature = feature/np.linalg.norm(feature)
            emb[idx] = feature
            
        np.save(out_path,emb)
