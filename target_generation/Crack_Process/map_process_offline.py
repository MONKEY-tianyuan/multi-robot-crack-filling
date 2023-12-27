''''
    this code generates the crack map as datasets.
    crack mode: gaussian, union

    author: Tianyuan Zheng
    date: 12.08.2023
'''

import torch
import numpy as np
import cv2
from torchvision.transforms import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
import copy

from skimage.morphology import skeletonize
from skimage.util import invert


intput_dir = 'target_generation/Data/CrackMaps'
output_dir = 'target_generation/Data/CrackMaps_MR'

def spdist(p,Ps):
    if p.dtype == 'int':
        p_float = p.astype(np.float32)
    if Ps.dtype == 'int':
        Ps_float = Ps.astype(np.float32)
    return np.sqrt(np.sum((p_float-Ps_float)**2,1))

def endpoints_intpoint_of_skeleton(skeleton):
    W,H = skeleton.shape
    img = copy.deepcopy(skeleton)
    endpoints = []
    intpoints=[]
    x,y = np.where(img>0)

    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    
    for i,j in zip(x,y):
        image_conv = img[i-1:i+2,j-1:j+2]
        if np.sum(np.multiply(image_conv,kernel)) == 1:
            endpoints.append([i,j])
            
            
        elif np.sum(np.multiply(image_conv,kernel)) >2 and np.sum(np.multiply(image_conv,kernel)) < 9:
            
            intpoints.append([i,j])
            

    endpoints = np.asarray(endpoints)
    intpoints = np.asarray(intpoints)
    return endpoints,intpoints

    

#image = os.path.join(intput_dir,'Gaussian1','myCrackGauss_s5_35_.png')
image = os.path.join(intput_dir,'Uniform','myCrack6_90_2.png')
# img = Image.open(image)
# transform = transforms.Compose([transforms.PILToTensor()])
# img = transform(img)[0]
# crack = torch.where(img==0)
# crack_range = [min(crack[0]), max(crack[0]), min(crack[1]), max(crack[1])]
# crack_block = img[crack_range[0]:crack_range[1],crack_range[2]:crack_range[3]]
# print(crack_block.shape)
img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
print(img.shape)

_,img = cv2.threshold(img,127,255,0)
img = cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

img = invert(img)
img = cv2.dilate(img,kernel=np.ones([5,5],np.uint8))
# plt.imshow(img)
# plt.show()
skeleton = skeletonize(img)
print(skeleton)

endpoints,intpoints = endpoints_intpoint_of_skeleton(skeleton)
print('endpoints',endpoints)
print('intpoints',intpoints)

col_anomaly = np.zeros([1])

while col_anomaly.size != 0:

    endpoint_sort_ind = np.argsort(endpoints,axis=0)
    endpoints_row_sort = endpoints[endpoint_sort_ind[:,0],:]

    print('endpoints_row_sort',endpoints_row_sort)
    # print('endpoints_col_sort',endpoints_col_sort)

    endpoints_row_diff1 = np.diff(endpoints_row_sort[:,0])
    endpoints_row_diff2 = np.diff(endpoints_row_sort[:,1])
    endpoints_row_dist = np.sqrt(endpoints_row_diff1**2 +  endpoints_row_diff2**2)


    # print('row_diff1',endpoints_row_diff1)
    # print('row_diff2',endpoints_row_diff2)
    print('row_dist',endpoints_row_dist)
    row_anomaly=np.where(endpoints_row_dist<15)[0]
    

    for anomaly_endpoint_ind in row_anomaly:
        anomaly_endpoint = endpoints_row_sort[anomaly_endpoint_ind]
        dist_tmp = spdist(anomaly_endpoint,intpoints)
        anomaly_intpoint = intpoints[np.argmin(dist_tmp)]
        print('anomaly_endpoint',anomaly_endpoint)
        print('anomaly_intpoint',anomaly_intpoint)
        x_left = anomaly_endpoint[0]
        x_right = anomaly_intpoint[0]
        y_left = anomaly_endpoint[1]
        y_right = anomaly_intpoint[1]
        if x_left > x_right:
            x_tmp = x_left
            x_left = x_right
            x_right = x_tmp
        if y_left > y_right:
            y_tmp = y_left
            y_left = y_right
            y_right = y_tmp
        skeleton[np.ix_(range(x_left,x_right+1),range(y_left,y_right+1))] = 0

    endpoints_row_sort = np.delete(endpoints_row_sort,row_anomaly,0)

    #column detect anomaly
    endpoint_sort_ind = np.argsort(endpoints_row_sort,axis=0)
    endpoints_col_sort = endpoints[endpoint_sort_ind[:,1],:]




    endpoints_col_diff1 = np.diff(endpoints_col_sort[:,0])
    endpoints_col_diff2 = np.diff(endpoints_col_sort[:,1])
    endpoints_col_dist = np.sqrt(endpoints_col_diff1**2+endpoints_col_diff2**2)
    col_anomaly = np.where(endpoints_col_dist<15)[0]
    
    for anomaly_endpoint_ind in col_anomaly:
        anomaly_endpoint = endpoints_col_sort[anomaly_endpoint_ind]
        dist_tmp = spdist(anomaly_endpoint,intpoints)
        anomaly_intpoint = intpoints[np.argmin(dist_tmp)]
        print('anomaly_endpoint',anomaly_endpoint)
        print('anomaly_intpoint',anomaly_intpoint)
        x_left = anomaly_endpoint[0]
        x_right = anomaly_intpoint[0]
        y_left = anomaly_endpoint[1]
        y_right = anomaly_intpoint[1]
        if x_left > x_right:
            x_tmp = x_left
            x_left = x_right
            x_right = x_tmp
        if y_left > y_right:
            y_tmp = y_left
            y_left = y_right
            y_right = y_tmp
        skeleton[np.ix_(range(x_left,x_right+1),range(y_left,y_right+1))] = 0

    endpoints_col_sort = np.delete(endpoints_col_sort,col_anomaly,0)

endpoints,intpoints = endpoints_intpoint_of_skeleton(skeleton)

print('col_dist',endpoints_col_dist)

print('endpoint size',endpoints.shape)
plt.imshow(skeleton,origin='lower')
plt.scatter(endpoints[:,1],endpoints[:,0])
plt.scatter(intpoints[:,1],intpoints[:,0])


plt.show()
