
from inference import OUTS
from inference import detection_threshold
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import albumentations as A
import cv2

import torch
import cv2
import numpy as np
import os
import glob as glob
import random
from xml.etree import ElementTree as et
from config import CLASSES
DIR_TRAIN = './test'
train_images = glob.glob(f"{DIR_TRAIN}/*") 
train_images=list(set([i[:-4]+".jpg" for i in train_images]))
boxes=[[] for i in range(len(train_images))]
print(boxes)
#%%
for i in range(len(train_images)):
    
    labels = []
    image_name = train_images[i].split('/')[-1].split("\\")[1]
    
    annot_file_path=train_images[i][:-4]+".xml"
    image = cv2.imread(train_images[i])
    img=image 
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    tree = et.parse(annot_file_path)
    root = tree.getroot()
    for member in root.findall('object'):
        
        labels.append(CLASSES.index(member.find('name').text))
        
        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)
        boxes[i].append([xmin,ymin,xmax,ymax,member.find('name').text])

print(boxes[1])
#%%

def calc_iou( gt_bbox, pred_bbox):
    
    x_min_gt,y_min_gt,x_max_gt,y_max_gt= gt_bbox
    x_min_p,y_min_p,x_max_p,y_max_p= pred_bbox
    
    if x_min_p>x_max_gt or y_min_p>y_max_gt:
        return 0.0 
    if x_max_p<x_min_gt or y_max_p<y_min_gt:
        return 0.0 
    
        
    
    
    GT_bbox_area = (x_max_gt-x_min_gt) * (y_max_gt-y_min_gt)
    Pred_bbox_area =(x_max_p-x_min_p ) * (y_max_p-y_min_p)
    
    
    x_bottom_left =np.max([x_min_gt,x_min_p])
    y_bottom_left = np.max([y_min_gt,y_min_p])
    x_top_right = np.min([x_max_gt,x_max_p])
    y_top_right = np.min([y_max_gt,y_max_p])
    
    intersection_area = (x_top_right-x_bottom_left) * (y_top_right-y_bottom_left)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area
#%%
def get_single_image_results(gt_boxes, outputs, iou_thr):
    
    pred_boxes = outputs[0]['boxes'].cpu().data.numpy()
    scores = outputs[0]['scores'].cpu().data.numpy()
    
    pred_boxes = pred_boxes[scores >= detection_threshold].astype(np.int32)
    for i in range(len(gt_boxes)):
        gt_boxes[i]=gt_boxes[i][:-1]
    for i in range(len(pred_boxes)):
        pred_boxes[i]=pred_boxes[i][:]
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            
            iou= calc_iou(gt_box, pred_box)
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

def get_single_image_results_1(gt_boxes,outputs,iou_thr):
    pred_boxes = outputs[0]['boxes'].cpu().data.numpy()
    scores = outputs[0]['scores'].cpu().data.numpy()
    
    pred_boxes = pred_boxes[scores >= detection_threshold].astype(np.int32)
    pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()][:len(pred_boxes)]
    
    tp=0
    fp=0 
    fn=0 
    
    gt_labels=[]
    for i in range(len(gt_boxes)):
        gt_labels.append(gt_boxes[i][-1])
        gt_boxes[i]=gt_boxes[i][:-1]
    is_classified=[1 for i in range(len(gt_labels))]
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    
    
    
    
    #Create a dictionary 
    dictionary=dict(zip(CLASSES,[0 for i in range(len(CLASSES))]))
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            
            iou= calc_iou(gt_box, pred_box) 
            
            if iou >iou_thr:
                is_classified[igb]=0
                if pred_classes[ipb]==gt_labels[igb]:
                    tp+=1 
                else:
                    fp+=1 
    
    fn=sum(is_classified)
    if tp==0:
        precision,recall=0,0 
    else:
        precision= (tp/(tp+fp))
        recall= (tp/(tp+fn))
    return precision,recall,tp,fp,fn
            
precision=[]
recall=[]
tp_all=0
fp_all=0
fn_all=0
for i in range(len(OUTS)):
    p,r,tp,fp,fn=(get_single_image_results_1(boxes[i], OUTS[i], 0.5))
    precision.append(p)
    recall.append(r)
    tp_all+=tp 
    fp_all+=fp 
    fn_all+=fn 
    
    
precision=(tp_all/(tp_all+fp_all))
recall=(tp_all/(tp_all+fn_all))

F2_score=(2*precision*recall)/(recall+precision)

print(f"Overall Precision : {precision}")
print(f"Overall Recall : {recall}")
print(F2_score)

    
    