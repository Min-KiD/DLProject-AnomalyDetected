import argparse
import torch
from torch.nn import init
import torch.nn as nn
import math
from functools import partial

import glob
import numpy as np
import os

from transform import ToTensor, Normalize
from models import Learner, resnet50, resnet101, resne152

from PIL import Image, ImageFilter, ImageOps, ImageChops
import random
import numbers
import time
import cv2
from matplotlib import pyplot as plt
#import pdb

def generate_model():
    model = resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            input_channels=3,
            output_layers=[])
    

    model = model.cuda()
    model = nn.DataParallel(model)

    return model

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection')
    parser.add_argument('file_path', type=str, help='Path to the video file')
    args = parser.parse_args()
    file_path = args.file_path

    # Check if the directory exists in /kaggle/working/ and create it if not
    save_directory = '/kaggle/working/' + os.path.basename(file_path)[:-4]
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

    # Define the path for saving the images in the created directory
    save_name = save_directory + '/%05d.jpg'

    # Command to extract frames and save in the created directory
    os.system('ffmpeg -i %s -r 25 -q:v 2 -vf scale=320:240 %s' % (file_path, save_name))

    model = generate_model() # feature extrctir
    classifier = Learner().cuda() # classifier
    # classifier = nn.DataParallel(classifier)

    checkpoint = torch.load('/kaggle/working/RGB_Kinetics_16f.pth')
    model.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load('/kaggle/working/model.pth')
    classifier.load_state_dict(checkpoint['net'])

    model.eval()
    classifier.eval()

    video = file_path.split('/')[-1][:-4]
    path = '/kaggle/working/'+ video + '/*'
    save_path = '/kaggle/working/'+ video +'_result'
    img = glob.glob(path)
    img.sort()

    segment = len(img)//16
    x_value =[i for i in range(segment)]

    inputs = torch.Tensor(1, 3, 16, 240, 320)
    x_time = [jj for jj in range(len(img))]
    y_pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for num, i in enumerate(img):
        if num < 16:
            inputs[:,:,num,:,:] = ToTensor(1)(Image.open(i))
            cv_img = cv2.imread(i)
            # print(cv_img.shape)
            h,w,_ =cv_img.shape
            cv_img = cv2.putText(cv_img, 'FPS : 0.0, Pred : 0.0', (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (173, 216, 230), 2)
        else:
            inputs[:,:,:15,:,:] = inputs[:,:,1:,:,:]
            inputs[:,:,15,:,:] = ToTensor(1)(Image.open(i))
            inputs = inputs.cuda()
            start = time.time()
            output, feature = model(inputs)
            feature = F.normalize(feature, p=2, dim=1)
            #print(feature.shape)
            out = classifier(feature)
            y_pred.append(out.item())
            end = time.time()
            FPS = str(1/(end-start))[:5]
            out_str = str(out.item())[:5]
            #print(len(x_value)/len(y_pred))
                    
            cv_img = cv2.imread(i)
            cv_img = cv2.putText(cv_img, 'FPS :'+FPS+' Pred :'+out_str, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (173, 216, 230), 2)
            if out.item() > 0.45:
                cv_img = cv2.rectangle(cv_img,(0,0),(w,h), (0,0,255), 3)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        path = save_path+'/'+os.path.basename(i)
        cv2.imwrite(path, cv_img)

    os.system('ffmpeg -i "%s" "%s"'%(save_path+'/%05d.jpg', save_path+'.mp4'))
    plt.plot(x_time, y_pred)
    plt.savefig(save_path+'.png', dpi=300)
    #plt.cla()

if __name__ == '__main__':
    main()
