import torch
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
import cv2
import os
from matplotlib import pyplot as plt
import copy
import math
import shapely
import geopandas as gd
from shapely import buffer,Point,LineString,MultiPolygon

from link import Link

class Crack_Graph():
    def __init__(self,
                 path=None,
                 ) -> None:
        super(Crack_Graph,self).__init__()
        self.botD = 48
        self.footD=7
        self.sensD = 4.5*12
        self.r1 = self.inpxMap(self.botD/2)
        self.a = self.inpxMap(self.footD/2)
        self.s = self.inpxMap(self.sensD/2)

        self.image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        self.skeleton = None
        self.skeleton_modified = None
        self.fp = None
        self.h_fp = []
        self.det = []

        self.link = None

        self.h_Line_list = []
        self.h_Partial_Crack_list = []
        self.h_Node_list = []
        self.h_Node_Order_list = []

        self.Line_list = []
        self.Partial_Crack_list = []
        self.Node_list = []
        self.Node_Order_list = []

        self.Node_set = None
        self.Polybuffer = None

    def search_overlap(self):
        num_partial_crack = len(self.Partial_Crack_list)
        overlap_mat = np.zeros([num_partial_crack,num_partial_crack])
        overlap_idx_pair = []
        overlap_area = []

        for ind1,geom1 in enumerate(self.Polybuffer.geometry):
            for ind2,geom2 in enumerate(self.Polybuffer.geometry):
                if ind2>ind1:
                    overlaps = geom1.overlaps(geom2)

                    if overlaps:
                        overlap_idx_pair.append([ind1,ind2])


        for pair in (overlap_idx_pair):
            overlap_area_tmp = self.Polybuffer.geometry[pair[0]].intersection(self.Polybuffer.geometry[pair[1]])
            overlap_area.append(overlap_area_tmp)
            self.Polybuffer.geometry[pair[0]] -= overlap_area_tmp
            self.Polybuffer.geometry[pair[1]] -= overlap_area_tmp
        
            self.Polybuffer = self.Polybuffer.union(overlap_area_tmp)
            

        print(1)

            
    def Minkows_sum(self):
        Poly = []
        for x,y in zip(self.link.x,self.link.y):
            poly = buffer(LineString(np.concatenate(([x],[y]),axis=0).T),self.a)
            
            px,py = poly.boundary.xy
            
            poly = poly.simplify(tolerance=1)
            Poly.append(poly)

        self.Polybuffer = gd.GeoSeries(Poly)

    def endpoint_reduce(self):
        endPoints = copy.deepcopy(self.Node_set)
        tmp_endPoints = copy.deepcopy(self.Node_set)
        endNodes = []
        e = 0
        while tmp_endPoints.size!=0:
            endlogi = np.where((np.asarray(self.Node_list).reshape([-1,2])==tmp_endPoints[e]).all(axis=1),1,0).reshape([-1,2])
            endlogi = np.sum(endlogi,axis=-1)
            Testpoints = tmp_endPoints
            for ind,logi in enumerate(endlogi):
                if logi:
                    del_idx_1 = np.where((Testpoints==self.Node_list[ind][:2]).all(axis=1))
                    Testpoints = np.delete(Testpoints,del_idx_1,axis=0)
                    del_idx_2 = np.where((Testpoints==self.Node_list[ind][2:]).all(axis=1))
                    Testpoints = np.delete(Testpoints,del_idx_2,axis=0)
            
            if np.all(self.spdist(tmp_endPoints[e],Testpoints)>self.a):
                endNodes.append(tmp_endPoints[e])
                
            tmp_endPoints = np.delete(tmp_endPoints,e,axis=0)

        print(1)  
   

    def connect_Crack_Graph(self):
        #connect the crack graph
        detind = []
        tmp_Node_list = copy.deepcopy(self.Node_list)
        tmp_Partial_Crack_list = copy.deepcopy(self.Partial_Crack_list)
        self.link = Link(branch_len=len(self.Line_list))

        intt = 0
        self.link.x[intt] = self.Line_list[intt][:,0]
        self.link.y[intt] = self.Line_list[intt][:,1]

        tmp_Node_list[intt] = np.zeros_like(tmp_Node_list[intt])
        intt2 = np.arange(intt+1,len(self.Line_list))
        indt = intt

        while(intt2.size!=0):

            tmpdist = self.spdist([self.link.x[intt][-1],self.link.y[intt][-1]],np.asarray(tmp_Node_list)[:,:2])
            inst = np.where(tmpdist <= self.a)[0]

            if inst.size!=0:
                inst = np.argmin(tmpdist)
                v1 = tmp_Partial_Crack_list[indt][0] - tmp_Partial_Crack_list[indt][-1]
                v2 = self.Partial_Crack_list[intt][0] - self.Partial_Crack_list[intt][-1]
                v3 = self.Partial_Crack_list[inst][0] - self.Partial_Crack_list[inst][-1]
                ang = self.ab2v(v1,v3)
                indt = inst
                ang2 = self.ab2v(v2,v3)

            if inst.size!=0 and not(ang<45.0/180*np.pi or ang>(360-45.0)/180*np.pi) and not(ang2<45.0/180*np.pi or ang2>(360-45.0)/180*np.pi):
                d = np.argmin(tmpdist)
                self.link.x[intt] = np.concatenate((self.link.x[intt],self.Line_list[d][:,0]))
                self.link.y[intt] = np.concatenate((self.link.y[intt],self.Line_list[d][:,1]))
                self.Node_list[intt][2:] = [self.link.x[intt][-1],self.link.y[intt][-1]]
                self.Partial_Crack_list[intt] = np.concatenate((self.Partial_Crack_list[intt],self.Partial_Crack_list[d]))
                tmp_Node_list[d] = np.zeros_like(tmp_Node_list[d])
                detind.append(d)
                intt2 = np.delete(intt2,intt2 == d)
            else:
                tmpdist = self.spdist([self.link.x[intt][-1],self.link.y[intt][-1]],np.asarray(tmp_Node_list)[:,2:4])
                inst = np.where(tmpdist <= self.a)[0]
                if inst.size!=0:
                    inst = np.argmin(tmpdist)
                    v1 = tmp_Partial_Crack_list[indt][0] - tmp_Partial_Crack_list[indt][-1]
                    v2 = self.Partial_Crack_list[intt][0] - self.Partial_Crack_list[intt][-1]
                    v3 = self.Partial_Crack_list[inst][0] - self.Partial_Crack_list[inst][-1]
                    ang = self.ab2v(v1,v3)
                    indt = inst
                    ang2 = self.ab2v(v2,v3)
                if inst.size!=0 and not(ang<45.0/180*np.pi or ang>(360-45.0)/180*np.pi) and not(ang2<45.0/180*np.pi or ang2>(360-45.0)/180*np.pi):
                    d = np.argmin(tmpdist)
                    self.link.x[intt] = np.concatenate((self.link.x[intt],np.flip(self.Line_list[d][:,0])))
                    self.link.y[intt] = np.concatenate((self.link.y[intt],np.flip(self.Line_list[d][:,1])))
                    self.Node_list[intt][2:] = [self.link.x[intt][-1],self.link.y[intt][-1]]
                    self.Partial_Crack_list[intt] = np.concatenate((self.Partial_Crack_list[intt],np.flip(self.Partial_Crack_list[d],axis=0)))
                    tmp_Partial_Crack_list[d] = np.flip(tmp_Partial_Crack_list[d],axis=0)
                    tmp_Node_list[d] = np.zeros_like(tmp_Node_list[d])
                    detind.append(d)
                    intt2 = np.delete(intt2,intt2 == d)
                else:
                    intt = intt2[0]
                    indt = intt
                    self.link.x[intt] = np.concatenate((self.link.x[intt],self.Line_list[intt][:,0]))
                    self.link.y[intt] = np.concatenate((self.link.y[intt],self.Line_list[intt][:,1]))
                    self.Node_list[intt][2:] = [self.link.x[intt][-1],self.link.y[intt][-1]]
                    tmp_Node_list[intt] = np.zeros_like(tmp_Node_list[intt])
                    intt2 = np.delete(intt2,intt2 == intt)
            
            #print(1)
            if np.any(np.asarray(tmp_Node_list)):
                continue
            else:
                break
        
        #update variables
        for d in detind:
            del self.Node_list[d]
            del self.Partial_Crack_list[d]
            self.fp = np.delete(self.fp, d)
           
        
        self.link.x = [x for x in self.link.x if len(x)!=0]
        self.link.y = [y for y in self.link.y if len(y)!=0]

        if len(self.h_Line_list)!=0:
            self.fp = np.concatenate((self.fp,np.asarray(self.h_fp)))
            for line,node,partial_crack in zip(self.h_Line_list,self.h_Node_list,self.h_Partial_Crack_list):
                self.Node_list.append(node)
                self.Partial_Crack_list.append(partial_crack)

                self.link.x.append(np.flip(line[:,0]))
                self.link.y.append(np.flip(line[:,1]))
        
        self.Node_set = np.unique(np.asarray(self.Node_list).reshape([-1,2]),axis=1)

            



    

    def acute_angle_update_list(self):
        
        for change_radius in self.det:
            self.h_Line_list.append(self.Line_list[change_radius])
            self.h_Partial_Crack_list.append(self.Partial_Crack_list[change_radius])
            self.h_Node_list.append(self.Node_list[change_radius])
            self.h_Node_Order_list.append(self.Node_Order_list[change_radius])
            self.h_fp.append(self.fp[change_radius])

            del self.Line_list[change_radius]
            del self.Partial_Crack_list[change_radius]
            del self.Node_list[change_radius]
            del self.Node_Order_list[change_radius]
            self.fp = np.delete(self.fp,change_radius)


    def acute_angle_process(self,Partial_Crack_list):
        for ind, partial_crack in enumerate(Partial_Crack_list):
            if self.totalLength(partial_crack) > 2*math.sqrt(2)*self.a:
                aRan = gd.GeoDataFrame([Point(partial_crack[0]).buffer(distance=self.a),Point(partial_crack[-1]).buffer(distance=self.a)],\
                                       columns=['geometry'])
                aRan['geometry'] = aRan.geometry.unary_union
                crack_not_inter = []
                int_flag = 1
                for partial_crack_pt in partial_crack:
                    
                    inter = shapely.intersection(aRan,Point(partial_crack_pt))
                    if np.isnan(inter['geometry'].x[0]):
                        crack_not_inter = partial_crack_pt
                        int_flag = 0
                        break
                
                if int_flag == 0:
                    # meaning there is crack outside endpoint circles
                    mov = partial_crack - crack_not_inter
                elif int_flag == 1:
                    mov = partial_crack - partial_crack[0]
            else:
                mov = partial_crack - partial_crack[0]
            
            v1 = [0,1]
            v2 = mov[-1,:]
            ang = self.ab2v(v1,v2)

            rotv = np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
            rv2 = np.dot(rotv,mov.T).T
            peakPts_idx = [np.argmin(rv2[:,0]),np.argmax(rv2[:,0])]
            d = np.argmax(np.abs(rv2[peakPts_idx][0]))
            peakPts_idx = peakPts_idx[d]

            v1 = -rv2[peakPts_idx]
            v2 = rv2[-1] +v1

            ang = self.ab2v(v1,v2)/2

            if ang<45.0/180*np.pi or ang>(360-45.0)/180*np.pi:
                self.fp[ind] = np.tan(ang) * self.a
                self.det.append(ind)

        return self.det


    
    def crack_fill_radius_init(self,Line_list):
        line_num = len(Line_list)
        self.fp = self.a*np.ones([line_num])
        return self.fp
    
    def partition_crack(self):
        endpoints, intpoints = self.extract_vertice_and_edge()
        endpoint_list = copy.deepcopy(endpoints)
        intpoint_list = copy.deepcopy(intpoints)
        skeleton = self.skeleton_modified
        
        Partial_Crack_list_diff = []
        endpoints_diff = []
        intpoints_diff = []

        while endpoints.size != 0:
            start_pt = endpoints[0,:]
            endpoints = np.delete(endpoints,0,axis=0)
            skeleton[start_pt[0],start_pt[1]] = 0
            startx,starty = start_pt
            image_conv = self.conv_block_with_padding(skeleton,start_pt)
            i,j = np.where(image_conv==1)
            Partial_Crack = np.array([start_pt,[startx-1+i[0],starty-1+j[0]]])

            detect_flag = 0

            while detect_flag == 0:
                tmp_x,tmp_y = Partial_Crack[-1,:]
                skeleton[tmp_x,tmp_y] = 0
                image_conv = self.conv_block_with_padding(skeleton,[tmp_x,tmp_y])
                image_conv_val = self.conv2D_from_block(image_conv)

                if image_conv_val == 1:
                    i,j = np.where(image_conv==1)
                    if np.where((intpoints==[i[0],j[0]]).all(axis=1))[0].size!=0:
                        self.Partial_Crack_list.append(Partial_Crack)

                        partial_len,_ = Partial_Crack.shape
                        partial_len -= 1
                        index = np.asarray(np.round(np.linspace(0,partial_len,24)),dtype=np.int32)
                        if index[-1] != partial_len:
                            index[-1] = partial_len
                        if index[0] != 0:
                            index[0] = 0
                        line = np.zeros([24,2])

                        for i in range(24):
                            line[i] = Partial_Crack[index[i]]
                        
                        self.Line_list.append(line)

                        self.Node_list.append([start_pt[0],start_pt[1],tmp_x-1+i[0],tmp_y-1+j[0]])
                        intpoints = np.delete(intpoints,np.where((intpoints==[i,j]).all(axis=1))[0],0)
                        detect_flag = 1
                        Partial_Crack = []
                    else:
                        crack_extend = np.array([[tmp_x-1+i[0],tmp_y-1+j[0]]])
                        Partial_Crack = np.concatenate((Partial_Crack,crack_extend),axis=0)
                    
                
                elif image_conv_val == 0:
                    #detect if this is an endpoint
                    if np.where((endpoints==[tmp_x,tmp_y]).all(axis=1))[0].size!=0 :
                        self.Partial_Crack_list.append(Partial_Crack)

                        partial_len,_ = Partial_Crack.shape
                        partial_len -= 1
                        index = np.asarray(np.round(np.linspace(0,partial_len,24)),dtype=np.int32)
                        if index[-1] != partial_len:
                            index[-1] = partial_len
                        if index[0] != 0:
                            index[0] = 0
                        line = np.zeros([24,2])

                        for i in range(24):
                            line[i] = Partial_Crack[index[i]]
                        
                        self.Line_list.append(line)
                            
                        self.Node_list.append([start_pt[0],start_pt[1],tmp_x,tmp_y])
                        endpoints = np.delete(endpoints,np.where((endpoints==[tmp_x,tmp_y]).all(axis=1))[0],0)
                        detect_flag = 1
                        Partial_Crack = []
                    elif np.where((intpoints==[tmp_x,tmp_y]).all(axis=1))[0].size!=0 :
                        self.Partial_Crack_list.append(Partial_Crack)

                        partial_len,_ = Partial_Crack.shape
                        partial_len -= 1
                        index = np.asarray(np.round(np.linspace(0,partial_len,24)),dtype=np.int32)
                        if index[-1] != partial_len:
                            index[-1] = partial_len
                        if index[0] != 0:
                            index[0] = 0
                        line = np.zeros([24,2])

                        for i in range(24):
                            line[i] = Partial_Crack[index[i]]
                        
                        self.Line_list.append(line)

                        self.Node_list.append([start_pt[0],start_pt[1],tmp_x,tmp_y])
                        intpoints = np.delete(intpoints,np.where((intpoints==[tmp_x,tmp_y]).all(axis=1))[0],0)
                        detect_flag = 1
                        Partial_Crack = []
                    else:
                        Partial_Crack_list_diff.append(Partial_Crack)
                        #Node_list = Node_list.append([start_pt[0],start_pt[1],tmp_x,tmp_y])
                        endpoints_diff = endpoints_diff.append([tmp_x,tmp_y])
                        detect_flag = 1
                        Partial_Crack = []

                elif image_conv_val>=2 and image_conv_val<9:
                    #detect if this is an intersection point
                    if np.where((intpoints==[tmp_x,tmp_y]).all(axis=1))[0].size!=0 :
                        self.Partial_Crack_list.append(Partial_Crack)

                        partial_len,_ = Partial_Crack.shape
                        partial_len -= 1
                        index = np.asarray(np.round(np.linspace(0,partial_len,24)),dtype=np.int32)
                        if index[-1] != partial_len:
                            index[-1] = partial_len
                        if index[0] != 0:
                            index[0] = 0
                        line = np.zeros([24,2])

                        for i in range(24):
                            line[i] = Partial_Crack[index[i]]
                        
                        self.Line_list.append(line)

                        self.Node_list.append([start_pt[0],start_pt[1],tmp_x,tmp_y])
                        intpoints = np.delete(intpoints,np.where((intpoints==[tmp_x,tmp_y]).all(axis=1))[0],0)
                        detect_flag = 1
                        Partial_Crack = []
                    else:
                        Partial_Crack_list_diff.append(Partial_Crack)
                        
                        intpoints_diff = intpoints_diff.append([tmp_x,tmp_y])
                        detect_flag = 1
                        Partial_Crack = []
        

        for intpoint in intpoints:
            intpoint_list = np.delete(intpoint_list,np.where((intpoint_list==intpoint).all(axis=1))[0],0)
            
        self.Node_set = np.concatenate((endpoint_list,intpoint_list),axis=0)

        for node in self.Node_list:
            start_id = np.where((self.Node_set==node[:2]).all(axis=1))[0]
            end_id = np.where((self.Node_set==node[2:]).all(axis=1))[0]
            self.Node_Order_list.append([start_id[0],end_id[0]])
        
        

        return Partial_Crack_list_diff


            

    def extract_vertice_and_edge(self):
        # skeletonize the image
        img = self.image
        _,img = cv2.threshold(img,127,255,0)
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        img = invert(img)
        skeleton = skeletonize(img)
        self.skeleton = skeleton

        #obtain endpoints and intersection points
        endpoints, intpoints = self.endpoints_intpoint_of_skeleton(skeleton)
        col_anomaly = np.zeros([1])

        #filter
        while col_anomaly.size != 0:
            endpoint_sort_ind = np.argsort(endpoints,axis=0)
            endpoints_row_sort = endpoints[endpoint_sort_ind[:,0],:]

            endpoints_row_diff1 = np.diff(endpoints_row_sort[:,0])
            endpoints_row_diff2 = np.diff(endpoints_row_sort[:,1])
            endpoints_row_dist = np.sqrt(endpoints_row_diff1**2 +  endpoints_row_diff2**2)

            row_anomaly=np.where(endpoints_row_dist<15)[0]
            
            for anomaly_endpoint_ind in row_anomaly:
                anomaly_endpoint = endpoints_row_sort[anomaly_endpoint_ind]
                dist_tmp = self.spdist(anomaly_endpoint,intpoints)
                anomaly_intpoint = intpoints[np.argmin(dist_tmp)]

                skeleton = self.set_value_block(skeleton,anomaly_endpoint,anomaly_intpoint)

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
                dist_tmp = self.spdist(anomaly_endpoint,intpoints)
                anomaly_intpoint = intpoints[np.argmin(dist_tmp)]
                
                skeleton = self.set_value_block(skeleton,anomaly_endpoint,anomaly_intpoint)

            endpoints_col_sort = np.delete(endpoints_col_sort,col_anomaly,0)

        #obtain endpoints and intersection points
        self.skeleton_modified = skeleton
        endpoints, intpoints = self.endpoints_intpoint_of_skeleton(skeleton)
        return endpoints, intpoints
    

    
    def endpoints_intpoint_of_skeleton(self,skeleton):
        W,H = skeleton.shape
        img = copy.deepcopy(skeleton)
        endpoints = []
        intpoints=[]
        x,y = np.where(img>0)
        
        for i,j in zip(x,y):
            image_conv_val = self.conv2D(img,[i,j])
            if image_conv_val == 1:
                endpoints.append([i,j])
                               
            elif image_conv_val >2 and image_conv_val < 9:
                intpoints.append([i,j])
                
        endpoints = np.asarray(endpoints)
        intpoints = np.asarray(intpoints)
        return endpoints,intpoints



    def conv2D(self,img,pos):
        img_conv = self.conv_block_with_padding(img,pos)

        kernel = np.array([[1,1,1],
                        [1,0,1],
                        [1,1,1]])
        
        return np.sum(np.multiply(img_conv,kernel))
    
    @staticmethod
    def conv2D_from_block(conv_block):
        img_conv = conv_block
        kernel = np.array([[1,1,1],
                        [1,0,1],
                        [1,1,1]])
        
        return np.sum(np.multiply(img_conv,kernel))

    @staticmethod
    def conv_block_with_padding(img,pos):
        W,H = img.shape
        i,j = pos
        if i == 0:
            if j == 0:
                conv_block = img[i:i+2,j:j+2]
                conv_block = np.concatenate((conv_block,np.zeros([1,2])),axis=0)
                conv_block = np.concatenate((np.zeros([3,1]),conv_block),axis=1)

            elif j == H-1:
                conv_block = img[i:i+2,j-1:j+1]
                conv_block = np.concatenate((np.zeros([1,2]),conv_block),axis=0)
                conv_block = np.concatenate((np.zeros([3,1]),conv_block),axis=1)
            
            else:
                conv_block = img[i:i+2,j-1:j+2]
                conv_block = np.concatenate((np.zeros([3,1]),conv_block),axis=1)
        elif i == W-1:
            if j == 0:
                conv_block = img[i-1:i+1,j:j+2]
                conv_block = np.concatenate((conv_block,np.zeros([1,2])),axis=0)
                conv_block = np.concatenate((conv_block,np.zeros([3,1])),axis=1)

            elif j == H-1:
                conv_block = img[i-1:i+1,j-1:j+1]
                conv_block = np.concatenate((np.zeros([1,2]),conv_block),axis=0)
                conv_block = np.concatenate((conv_block,np.zeros([3,1])),axis=1)
            
            else:
                conv_block = img[i-1:i+1,j-1:j+2]
                conv_block = np.concatenate((conv_block,np.zeros([3,1])),axis=1)
        else :
            if j == 0:
                conv_block = img[i-1:i+2,j:j+2]
                conv_block = np.concatenate((conv_block,np.zeros([1,2])),axis=0)

            elif j == H-1:
                conv_block = img[i-1:i+2,j-1:j+1]
                conv_block = np.concatenate((np.zeros([1,2]),conv_block),axis=0)            
            else:
                conv_block = img[i-1:i+2,j-1:j+2]
        return conv_block
    
    @staticmethod
    def ab2v(a,b):
        a = np.asarray(a,dtype=np.float32)
        b = np.asarray(b,dtype=np.float32)

        cos_angle = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        sin_angle = np.cross(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

        angle = np.arccos(np.abs(cos_angle))

        if cos_angle>0 and sin_angle<0:
            angle = -angle
        elif cos_angle<0 and sin_angle>0:
            angle = np.pi - angle
        elif cos_angle<0 and sin_angle<0:
            angle = np.pi + angle
        

        if angle<0:
            angle = 2*np.pi + angle

        return angle

    @staticmethod
    def spdist(p,Ps):
        p = np.asarray(p,dtype=np.float32)
        Ps = np.asarray(Ps,dtype=np.float32)
        return np.sqrt(np.sum((p-Ps)**2,1))
        
    @staticmethod
    def midP(P1,P2):
        return (P1+P2)/2
    
    @staticmethod
    def totalLength(Ps):
        return np.sqrt(np.sum(np.diff(Ps,axis=0)**2))
    
    @staticmethod
    def inpxMap(x):
        return int(x*25.4/2)
    
    @staticmethod
    def set_value_block(img,p1,p2,
                        value=0):
        x_left = p1[0]
        x_right = p2[0]
        y_left = p1[1]
        y_right = p2[1]
        if x_left > x_right:
            x_tmp = x_left
            x_left = x_right
            x_right = x_tmp
        if y_left > y_right:
            y_tmp = y_left
            y_left = y_right
            y_right = y_tmp
        img[np.ix_(range(x_left,x_right+1),range(y_left,y_right+1))] = value
        return img
    



