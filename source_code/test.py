# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:15:25 2022
@author: Herminarto Nugroho
Ini program untuk membuat dataframe dalam bentuk csv dari seluruh folder images,
 dengan terlebih dahulu mengecek nama-nama file yang sama dari seluruh folder 
 training left, right, disparity dan depth.
 
"""

import os
import pandas as pd

BASE_DIR = 'D:/STEREO CAMERA PROJECT/'
train_left = BASE_DIR+'train-left-image/'
train_right = BASE_DIR+'train-right-image/'
train_disparity = BASE_DIR+'train-disparity-map/'
train_depth = BASE_DIR+'train-depth-map/'

#[os.path.splitext(filename)[0] for filename in os.listdir(path)]

files_in_left = sorted(os.path.splitext(filename)[0] for filename in os.listdir(train_left))
files_in_right = sorted(os.path.splitext(filename)[0] for filename in os.listdir(train_right))
files_in_disparity = sorted(os.path.splitext(filename)[0] for filename in os.listdir(train_disparity))
files_in_depth = sorted(os.path.splitext(filename)[0] for filename in os.listdir(train_depth))

#images=[i for i in files_in_train if i in files_in_annotated]
images=[i for i in files_in_left if i in files_in_right if i in files_in_disparity if i in files_in_depth]

df = pd.DataFrame()
df['left']=[train_left+str(x)+str('.jpg') for x in images]
df['right']=[train_right+str(x)+str('.jpg') for x in images]
df['disparity']=[train_disparity+str(x)+str('.png') for x in images]
df['depth']=[train_depth+str(x)+str('.png') for x in images]
#df['labels']=[train_annotation+str(x) for x in images]

df.to_csv('D:/STEREO CAMERA PROJECT/datasets.csv', header=None)