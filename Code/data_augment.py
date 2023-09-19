import albumentations as A
import cv2

import torch
import cv2
import numpy as np
import os
import glob as glob
import random
from xml.etree import ElementTree as et

from albumentations.augmentations.transforms import (ChannelShuffle,CLAHE,ColorJitter,Downscale,Emboss,Equalize,FancyPCA,FromFloat,GaussNoise,ImageCompression,
InvertImg,ISONoise,JpegCompression,Lambda,MultiplicativeNoise,Normalize,PixelDropout,Posterize,RandomBrightness,RandomBrightnessContrast,RandomContrast,RandomFog,
RandomGamma,RandomGridShuffle,RandomRain,RandomShadow,RandomSnow,RandomSunFlare,RandomToneCurve,RGBShift,RingingOvershoot,Sharpen,Solarize,Spatter,Superpixels,ToFloat,ToGray,ToSepia,UnsharpMask)
from albumentations.augmentations.blur.transforms import AdvancedBlur,Blur,Defocus,GaussianBlur,GlassBlur,MedianBlur,MotionBlur,ZoomBlur


from albumentations.augmentations.geometric.transforms import Affine,ElasticTransform,Flip,HorizontalFlip,Perspective,PiecewiseAffine,ShiftScaleRotate,Transpose,VerticalFlip,GridDistortion
from albumentations.augmentations.crops.transforms import BBoxSafeRandomCrop,CenterCrop,Crop,CropAndPad,CropNonEmptyMaskIfExists,RandomCrop,RandomCropFromBorders,RandomCropNearBBox,RandomResizedCrop,RandomSizedBBoxSafeCrop,RandomSizedCrop
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout 
from albumentations.augmentations.geometric.resize import LongestMaxSize,RandomScale,Resize,SmallestMaxSize
from albumentations.augmentations.dropout.mask_dropout import MaskDropout 
from albumentations.core.transforms_interface import BasicTransform,DualTransform,ImageOnlyTransform,NoOp


import shutil


import matplotlib.pyplot as plt 



# box coordinates for xml files are extracted and corrected for image size given

    
#CLASSES = ['Atmospheric pressure limitation', 'Authorized Representative', 'Batch Code', 'Biological risks', 'Catalogue Number', 'Caution', 'Consult instructions', 'Contains Latex', 'Contains sufficient for -n- tests', 'Control', 'Date of Manufacture', 'Do not resterilize', 'Do not reuse', 'Do not use if package is damaged', 'Drops per milliliter', 'Fluid Path', 'For IVD perfomance evaluation only', 'Fragile Handle with care', 'Humidity Limitation', 'In vitro Diagnostic Medical device', 'Keep Dry', 'Keep away from sunlight', 'Liquid filter with pore size', 'Lower limit of temperature', 'Manufacturer', 'Negative Control', 'Non Pyrogenic', 'Non sterile', 'One way valve', 'Patient Number', 'Positive Control', 'Protect from heat radioactive sources', 'Sampling site', 'Sterile Fluid Path', 'Sterile', 'Sterilized using aseptic techniques', 'Sterilized using ethylene oxide', 'Sterilized using irradiation', 'Sterilized using steam', 'Temperature Limit', 'Upper limit of temperature', 'Use By Date',"glove pairs", "medical device", "powder free","Recyclable"]
CLASSES=["Workpiece"]
t1=[ChannelShuffle,CLAHE,ColorJitter,Downscale,Emboss,Equalize,FancyPCA,FromFloat,GaussNoise,ImageCompression,
InvertImg,ISONoise,JpegCompression,MultiplicativeNoise,Normalize,PixelDropout,Posterize,RandomBrightness,RandomBrightnessContrast,RandomContrast,RandomFog,
RandomGamma,RandomRain,RandomShadow,RandomSnow,RandomSunFlare,RandomToneCurve,RGBShift,RingingOvershoot,Sharpen,Solarize,Spatter,Superpixels,ToFloat,ToGray,ToSepia,UnsharpMask]


t2=[AdvancedBlur,Blur,Defocus,GaussianBlur,GlassBlur,MedianBlur,MotionBlur,ZoomBlur]


t3=[Affine,ElasticTransform,Flip,HorizontalFlip,Perspective,PiecewiseAffine,ShiftScaleRotate,Transpose,VerticalFlip,GridDistortion,BBoxSafeRandomCrop,CenterCrop,Crop,CropAndPad,CropNonEmptyMaskIfExists,RandomCrop,RandomCropFromBorders,RandomCropNearBBox,RandomResizedCrop,RandomSizedBBoxSafeCrop,RandomSizedCrop]
t4=[BasicTransform,DualTransform,ImageOnlyTransform,CoarseDropout]

def rigid_transforms(image):
    transform=A.Compose([random.choice(t1)(),Affine(),random.choice(t1)(),random.choice(t2)(),A.CoarseDropout()])
    
    image=transform(image=image)
    
    
    return image["image"]

def non_rigid_transforms(image,boxes):
    transform=A.Compose([random.choice(t1)(),random.choice(t2)(),Flip(),Affine()],bbox_params=A.BboxParams('pascal_voc'))
    image=transform(image=image,bboxes=boxes)
    
    img=image["image"]
    box=image["bboxes"]
    return img,box
    
    
DIR_TRAIN = './train'
train_images = glob.glob(f"{DIR_TRAIN}/*") 
train_images=list(set([i[:-4]+".jpg" for i in train_images]))
for i in range(len(train_images)):
    boxes = []
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
        boxes.append([xmin,ymin,xmax,ymax,member.find('name').text])
    
    
    try:
        for i in range(5):
            image=rigid_transforms(img)
            shutil.copy(annot_file_path,f"./final_augmented/{image_name}"+str(i)+".xml")
            name=f"./final_augmented/{image_name}"+str(i)+".xml"
            tree = et.parse(name)
            root = tree.getroot()
            root.find("filename").text=f"./final_augmented/{image_name}"+str(i)+".jpg"
            root.find("path").text=f"./final_augmented/{image_name}"+str(i)+".xml"
            cv2.imwrite(f"./final_augmented/{image_name}"+str(i)+".jpg",image)
            
        for i in range(5,10):
            image,box_main=non_rigid_transforms(img,boxes)
            
            print(box_main)
            shutil.copy(annot_file_path,f"./final_augmented/{image_name}"+str(i)+".xml")
            name=f"./final_augmented/{image_name}"+str(i)+".xml"
            tree = et.parse(name)
            root = tree.getroot()
            root.find("filename").text=f"./final_augmented/{image_name}"+str(i)+".jpg"
            root.find("path").text=f"./final_augmented/{image_name}"+str(i)+".xml"
            try:
                for k,member in enumerate(root.findall('object')):
                    box=box_main[k]
                    
                    member.find('name').text=box[-1]
                    member.find('bndbox').find('xmin').text=str(int(box[0]))
                    member.find('bndbox').find('ymin').text=str(int(box[1]))   
                    member.find('bndbox').find('xmax').text=str(int(box[2]))
                    member.find('bndbox').find('ymax').text=str(int(box[3]))
                    
                    cv2.rectangle(
                        image, 
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                        (0, 0, 255), 2
                    )
            except:
                pass
            
    
        tree.write(name)
        
        print(box)
        plt.imshow(image)
        cv2.imwrite(f"./final_augmented/{image_name}"+str(i)+".jpg",image)
    except:
            pass
