import torch
import numpy as np
import cv2
from torchvision.transforms import transforms
import os
from matplotlib import pyplot as plt
import copy

from crack_graph import Crack_Graph



intput_dir = 'target_generation/Data/CrackMaps'
output_dir = 'target_generation/Data/CrackMaps_MR'


image_path = os.path.join(intput_dir,'Uniform','myCrack6_90_2.png')


crack_model = Crack_Graph(image_path)

Partial_Crack_list_diff = crack_model.partition_crack()

plt.figure(num=1)
for partial_crack in crack_model.Partial_Crack_list:
    plt.plot(partial_crack[:,0],partial_crack[:,1])


radius_init = crack_model.crack_fill_radius_init(crack_model.Line_list)
#print(crack_model.ab2v([1,2],[3,4]))

det = crack_model.acute_angle_process(crack_model.Partial_Crack_list)

crack_model.acute_angle_update_list()

crack_model.connect_Crack_Graph()

crack_model.acute_angle_process(crack_model.Partial_Crack_list)

plt.figure(num=2)
for partial_crack in crack_model.Partial_Crack_list:
    plt.plot(partial_crack[:,0],partial_crack[:,1])

crack_model.endpoint_reduce()

crack_model.Minkows_sum()

crack_model.search_overlap()

plt.show()
