import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from glob import glob

from typing import List, Dict 



def open_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        img_path = data['image_filepath']
        
        img_traj = img_path.split("/")
        
        img_path = os.path.join(img_traj[0], img_traj[-1])
        tooth_info = data['tooth'][:]
    return img_path, tooth_info

def read_image(img_path):
    abs_img_path = os.path.abspath(img_path) # relative path makes error
    img = cv2.imread(abs_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def seperate_teeth_to_tooth_info(
        json_file: os.PathLike, dset_path: os.PathLike,
        dilate_kernel_size: int=21, img_padding: int=10
    ) -> List[Dict]: 
    """
        json file 에는 이미지의 경로, image_path와 각 teeth의 정보들이 주어진다. 
        하나의 tooth 이미지에 많은 teeth들이 있고 각 teeth에 대하여 충치 여부를 
        판단해야하기 때문에 이미지로부터 teeth들만을 추출해야함.
        
        [e.g.]
        json_file -> image_path -> tooth_image -> List[teeth_img1, teeth_img2, ..., teeth_imgN]
    
        Args: 
            json_file (os.PathLike): 대회에서 사용하는 json file 의 path
            dset_path (os.PathLike): image, json directory가 있는 상위 디렉토리의 경로
        Returns: (img_path, segmentation, decayed)
    """
    
    img_path, tooth_info = open_json(json_file)
    
    # read teeth image
    img_path = os.path.join(dset_path, img_path)
    img = read_image(img_path)
    
    # declare empty list
    teeth_dict_list = []
    
    ############### Deprecated ##################
    # teeth_imgs = []
    # decayeds = []
    # teeth_nums = []
    # categories = []
    #############################################

    for teeth in tooth_info:
        # segmentation 정보 추출
        segmentation = teeth['segmentation']
        segmentation = np.array(segmentation)
        
        # segmentaion 정보를 바탕으로 teeth 이미지를 masking 하여 teeth 부분만 남김.
        mask = np.zeros_like(img)
        mask = cv2.fillConvexPoly(mask, segmentation, color=(255, 255, 255))
        mask = cv2.dilate(mask, kernel=np.ones((dilate_kernel_size, dilate_kernel_size)), iterations=1)
        masked_teeth = cv2.bitwise_and(img, mask)
        
        # 추출한 teeth 부분을 정사각형 이미지로 만들기 위해 이미지의 width, height 계산 
        x_min, y_min = np.min(segmentation, axis=0)
        x_max, y_max = np.max(segmentation, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        # width, height 중 더 긴 길이에 padding을 추가하여 
        # 모두 0인 (teeth_img_size, teeth_img_size) 이미지 생성
        teeth_img_size = max(width, height) + (2 * img_padding)
        # teeth_img = np.zeros((teeth_img_size, teeth_img_size, 3), np.int32)
        teeth_img = np.zeros((teeth_img_size, teeth_img_size, 3), np.uint8)
        
        # height start, height end, width start, width end, locating teeth center
        h_s = (teeth_img_size//2) - (height//2) + img_padding
        w_s = (teeth_img_size//2) - (width//2) + img_padding
        
        # masking 된 이빨 이미지를 앞서 만들어둔 정사각형 이미지로 copy
        # segmentation 이 이미지 영역을 벗어나는 경우가 있어서 아래와 같이
        # 따로 width, height 를 복사할 이미지로부터 구하여 copy.
        masked_teeth_patch = masked_teeth[y_min:y_max, x_min:x_max]
        masked_height, masked_width, _ = masked_teeth_patch.shape
        
        teeth_img[h_s:h_s+masked_height, w_s:w_s+masked_width] = masked_teeth[y_min:y_max, x_min:x_max]
        
        category = img_path.split("/")[-1].split("_")[0]
        
        teeth_dict_list.append({
            "teeth_image": teeth_img,
            "teeth_num": teeth['teeth_num'],
            "teeth_position": category,
            "target": teeth['decayed']
        })
        
    return teeth_dict_list


def tooth_num_to_index(tooth_num):
    # 11,.., 18, 21,..,28, 31,..38, 41,..,48 -> 0, 1, 2, ..., 31
    return torch.LongTensor([((tooth_num // 10) - 1) * 8 + ((tooth_num % 10) - 1)])
    
def tooth_position_to_index(tooth_position):
    # Front -> 0, Left -> 1, Right -> 2, Lower -> 3, Upper -> 4
    tooth_position = tooth_position.lower()
    
    if tooth_position == "front":
        idx = 0
    elif tooth_position == "left":
        idx = 1
    elif tooth_position == "right":
        idx = 2
    elif tooth_position == "lower":
        idx = 3
    elif tooth_position == "upper":
        idx = 4
    else:
        raise RuntimeError(f"Error in src.utils.tooth_position_to_index: tooth_position: {tooth_position}")
    
    return torch.LongTensor([idx])
    
def tooth_num_to_one_hot_vec(tooth_num):
    tooth_num_idx = tooth_num_to_index(tooth_num)
    tooth_num_one_hot = F.one_hot(tooth_num_idx, num_classes=32).squeeze(0)
    return tooth_num_one_hot

def tooth_position_to_one_hot_vec(tooth_position):
    tooth_position_idx = tooth_position_to_index(tooth_position)
    tooth_position_one_hot = F.one_hot(tooth_position_idx, num_classes=5).squeeze(0)
    return tooth_position_one_hot