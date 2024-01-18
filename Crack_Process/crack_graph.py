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
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry import Polygon

from utils import *

from scipy.spatial import KDTree


class Crack_Graph():
    def __init__(self,
                 path=None,
                 ) -> None:
        super(Crack_Graph,self).__init__()
        self.botD = 48
        self.footD=7
        self.sensD = 2*12
        self.W = 3500
        self.H = 4000
        
        self.r1 = self.inpxMap(self.botD/2)
        self.a = self.inpxMap(self.footD/2)
        self.s = self.inpxMap(self.sensD/2)

        self.workSpace = Polygon([[0,0],[self.W,0],[self.W,self.H],[0,self.H]])

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
        self.NodeCan_set = None
        self.NodeRed_set = None
        self.Polybuffer = None
        self.Polybuffer_gd = None
        self.overlap_block = []
        self.endNodes = []
        
        self.vertice = []
        self.vgNE = []
        self.vgEE = []
        self.visibility = []
        self.edgeList = None
        self.edgeIndList = None
        
        self.objCrack = Polygon()
        self.scanSpace = None
        
    def run(self):
        
        self.partition_crack()
        self.crack_fill_radius_init(self.Line_list)
        self.acute_angle_process(self.Partial_Crack_list)
        self.acute_angle_update_list()
        self.connect_Crack_Graph()
        self.acute_angle_process(self.Partial_Crack_list)
        self.endpoint_reduce()
        self.Minkows_sum()
        self.search_overlap()
        self.endpoint_shorten()
        self.visibility_graph()
        self.poly_edges()
        return self.NodeRed_set, self.edgeList , self.edgeIndList,self.objCrack, self.scanSpace
    
    def poly_edges(self):
        for edges in self.vgNE:
            for line in edges:
                self.objCrack = self.objCrack.union(buffer(LineString(line.reshape([-1,2])),self.s))
        self.scanSpace = self.workSpace-self.objCrack
                
                
    def visibility_graph(self):
        for ind,poly in enumerate(self.Polybuffer):
            in_logi = np.asarray([poly.contains(Point(node)) for node in self.NodeRed_set])
            vertice_tmp = Vertice(inpoly_ind=np.where(in_logi)[0],node_set=self.NodeRed_set)

            link_tmp = np.concatenate([self.link.x[ind].reshape([-1,1]),self.link.y[ind].reshape([-1,1])],axis=1)
            mem = [np.where((link_tmp==node).all(axis=1))[0].size!=0   for node in vertice_tmp.node_pos]
            kdt = KDTree(link_tmp)
            d,_=np.array(kdt.query(vertice_tmp.node_pos))
            d = [x<self.a for x in d]
            mem = [m or x for m,x in zip(mem,d)]
            ind_mem = np.where(mem)[0]
            inpoly_ind = np.asarray([vertice_tmp.inpoly_ind[a] for a in ind_mem]).reshape([-1,1])
            node_pos = np.asarray([vertice_tmp.node_pos[a] for a in ind_mem]).reshape([-1,2])
            vertice_arr_tmp = np.concatenate([inpoly_ind,node_pos],axis=1)
            
            _,ind_dis = np.array(kdt.query(node_pos))
            vertice_arr_tmp = np.concatenate([ind_dis.reshape([-1,1]),vertice_arr_tmp],axis=1)
            vertice_arr_tmp = vertice_arr_tmp[vertice_arr_tmp[:,0].argsort()]
            S_NE = []
            S_EE = []
            if vertice_arr_tmp.shape[0] > 1:
                for i in range(vertice_arr_tmp.shape[0]-1):
                    S_NE.append(np.concatenate([vertice_arr_tmp[i,-2:],vertice_arr_tmp[i+1,-2:]]))
                    S_EE.append(np.array([vertice_arr_tmp[i,1],vertice_arr_tmp[i+1,1]]))
                    
            S_NE = np.asarray(S_NE)
            S_EE = np.asarray(S_EE)       
        
            if type(poly.boundary) is MultiLineString:
                mem1 = d
                ind_mem1 = np.where(mem1)[0]
                inpoly_ind1 = np.asarray([vertice_tmp.inpoly_ind[a] for a in ind_mem1]).reshape([-1,1])
                node_pos1 = np.asarray([vertice_tmp.node_pos[a] for a in ind_mem1]).reshape([-1,2])
                vertice_arr_tmp1 = np.concatenate([inpoly_ind1,node_pos1],axis=1)
            
                _,ind_dis1 = np.array(kdt.query(node_pos1))
                vertice_arr_tmp1 = np.concatenate([ind_dis1.reshape([-1,1]),vertice_arr_tmp1],axis=1)
                vertice_arr_tmp1 = vertice_arr_tmp1[vertice_arr_tmp1[:,0].argsort()]

                if vertice_arr_tmp1.shape[0] > 1:
                    for i in range(vertice_arr_tmp1.shape[0]-1):
                        NE_tmp1 = np.concatenate([vertice_arr_tmp1[i,-2:],vertice_arr_tmp1[i+1,-2:]])
                        EE_tmp1 = np.array([vertice_arr_tmp1[i,1],vertice_arr_tmp1[i+1,1]])
                        S_NE = np.concatenate([S_NE,NE_tmp1],axis=0)
                        S_EE = np.concatenate([S_EE,EE_tmp1],axis=0)
            
            not_mem = [not m for m in mem]
            ind_not_mem = np.where(not_mem)[0]
            inpoly_ind2 = np.asarray([vertice_tmp.inpoly_ind[a] for a in ind_not_mem]).reshape([-1,1])
            node_pos2 = np.asarray([vertice_tmp.node_pos[a] for a in ind_not_mem]).reshape([-1,2])
            _,ind_dis2 = np.array(kdt.query(node_pos2))
            node_pos2_tmp = np.asarray([vertice_arr_tmp[a,-2:] for a in ind_dis2]).reshape([-1,2])
            inpoly_ind2_tmp = np.asarray([vertice_arr_tmp[a,1] for a in ind_dis2]).reshape([-1,1])
            NE_tmp2 = np.concatenate([node_pos2,node_pos2_tmp],axis=1)
            EE_tmp2 = np.concatenate([inpoly_ind2,inpoly_ind2_tmp],axis=1)
            
            S_NE = np.concatenate([S_NE,NE_tmp2],axis=0)
            S_EE = np.concatenate([S_EE,EE_tmp2],axis=0)
                 
            self.vertice.append(vertice_tmp)
            self.vgNE.append(S_NE)
            self.vgEE.append(S_EE)
            
        for ind,poly in enumerate(self.Polybuffer):
            vis = Visbility()
            vis.visibility,vis.inter,vis.outer = line_of_sight2(obsv_node=self.vgNE[ind][:,:2],
                                                                    tgt_node=self.vgNE[ind][:,2:],
                                                                    ext_poly=poly)
            for j,node_vis in enumerate(vis.visibility):
                if not node_vis:
                    start = self.vgNE[ind][j,:2]
                    goal = self.vgNE[ind][j,2:]
                    startn = self.vgEE[ind][j,0]
                    goaln = self.vgEE[ind][j,1]
                    slen = self.NodeRed_set.shape[0]
                    waypoint,dist = pathfinder(start,goal,poly)
                    nn = waypoint[1:-1]
                    self.NodeRed_set = np.concatenate([self.NodeRed_set,nn],axis=0)
                    
                    if nn.size != 0:
                        self.vgNE[ind] = np.concatenate([self.vgNE[ind],np.concatenate([start,nn[0]]).reshape([-1,4])],axis=0)
                        self.vgEE[ind] = np.concatenate([self.vgEE[ind],np.array([startn,slen]).reshape([-1,2])],axis=0)
                        
                        if nn.shape[0]>1:
                            for k in range(1,nn.shape[0]):
                                self.vgNE[ind] = np.concatenate([self.vgNE[ind],np.concatenate([self.vgNE[ind][-1,2:],nn[k]]).reshape([-1,4])],axis=0)
                                self.vgEE[ind] = np.concatenate([self.vgEE[ind],np.array([self.vgEE[ind][-1,1],slen+k]).reshape([-1,2])],axis=0)
                        
                        self.vgNE[ind] = np.concatenate([self.vgNE[ind],np.concatenate([self.vgNE[ind][-1,2:],goal]).reshape([-1,4])],axis=0)
                        self.vgEE[ind] = np.concatenate([self.vgEE[ind],np.array([self.vgEE[ind][-1,1],goaln]).reshape([-1,2])],axis=0)
                    else:
                        self.vgNE[ind] = np.concatenate([self.vgNE[ind],np.concatenate([start,goal]).reshape([-1,4])],axis=0)
                        self.vgEE[ind] = np.concatenate([self.vgEE[ind],np.array([startn,goaln]).reshape([-1,2])],axis=0)
            
            self.vgNE[ind] = np.delete(self.vgNE[ind],np.where(vis.visibility==False)[0],axis=0)  
            self.vgEE[ind] = np.delete(self.vgEE[ind],np.where(vis.visibility==False)[0],axis=0)
        
        self.edgeIndList = self.vgEE[0]
        for i in range(1,len(self.vgEE)):
            self.edgeIndList = np.concatenate([self.edgeIndList,self.vgEE[i]],axis=0)
            
        self.edgeList = self.vgNE[0]
        for i in range(1,len(self.vgNE)):
            self.edgeList = np.concatenate([self.edgeList,self.vgNE[i]],axis=0)
        

                    
                    
    
    def endpoint_shorten(self):
        for ind,node in enumerate(self.NodeRed_set):
            aRan = shapely.buffer(Point(node),self.a/np.sqrt(2))
            logi_cell = [shapely.intersects(aRan,LineString(partial_crack)) for partial_crack in self.Partial_Crack_list]
            if np.sum(logi_cell)==1:
                crack_tmp = self.Partial_Crack_list[np.where(np.asarray(logi_cell)==True)[0][0]]
                endP_int_logi = np.asarray([shapely.intersects(aRan,Point(crack_tmp[0])), shapely.intersects(aRan,Point(crack_tmp[-1]))])
                if np.any(endP_int_logi):
                    int_poly = shapely.intersection(aRan,LineString(crack_tmp))
                    ext_poly = LineString(crack_tmp) - int_poly
                    ext_poly = ext_poly.xy
                    ext_poly = np.concatenate([np.asarray(ext_poly[0]).reshape([-1,1]),np.asarray(ext_poly[1]).reshape([-1,1])],axis=1)
                    if len(int_poly.xy[0])!=0:
                        self.Partial_Crack_list[np.where(np.asarray(logi_cell)==True)[0][0]] = ext_poly
                        ee = np.concatenate([crack_tmp[0].reshape([-1,2]),crack_tmp[-1].reshape([-1,2])],axis=0)
                        ddd = np.argmin(self.spdist(node,ee))
                        self.NodeRed_set[ind] = ee[ddd]
                
    
    def search_overlap(self):
        #num_partial_crack = len(self.Partial_Crack_list)
        overlap_idx_pair = []
        inter_centernode_list = []
        #inter_overlapnum_list = []

        for ind1,geom1 in enumerate(self.Polybuffer_gd.geometry):
            for ind2,geom2 in enumerate(self.Polybuffer_gd.geometry):
                if ind2>ind1:
                    overlaps = geom1.overlaps(geom2)

                    if overlaps:
                        overlap_idx_pair.append([ind1,ind2])

        if len(overlap_idx_pair) != 0:
            for pair in (overlap_idx_pair):
                overlap_area_tmp = self.Polybuffer_gd.geometry[pair[0]].intersection(self.Polybuffer_gd.geometry[pair[1]])
                overlap_block_tmp = GeoBlock(Poly=overlap_area_tmp,idx = pair)
                self.overlap_block.append(overlap_block_tmp)
                cx,cy = shapely.centroid(overlap_area_tmp).xy
                inter_centernode_list.append([cx[0],cy[0]])

            inter_centernode_list = np.asarray(inter_centernode_list,dtype=np.float64)
            self.NodeCan_set = np.concatenate([inter_centernode_list,self.endNodes],axis=0)
            self.NodeRed_set = self.NodeCan_set[0,:]
            for nodecan in self.NodeCan_set:
                if np.all(self.spdist(nodecan,self.NodeRed_set)>self.a):
                    self.NodeRed_set = np.concatenate([self.NodeRed_set.reshape([-1,2]),nodecan.reshape([-1,2])],axis=0)
        else:
            self.NodeRed_set = copy.deepcopy(self.endNodes)

            
    def Minkows_sum(self):
        Poly = []
        for x,y in zip(self.link.x,self.link.y):
            poly = buffer(LineString(np.concatenate(([x],[y]),axis=0).T),self.a)
            
            px,py = poly.boundary.xy
            
            poly = poly.simplify(tolerance=1)
            Poly.append(poly)

        self.Polybuffer = Poly
        self.Polybuffer_gd = gd.GeoSeries(Poly)

    def endpoint_reduce(self):
        endPoints = copy.deepcopy(self.Node_set)
        tmp_endPoints = copy.deepcopy(self.Node_set)
        
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
                self.endNodes.append(tmp_endPoints[e])
                
            tmp_endPoints = np.delete(tmp_endPoints,e,axis=0)


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

        for change_radius in self.det[::-1]:
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

        


    
    def crack_fill_radius_init(self,Line_list):
        line_num = len(Line_list)
        self.fp = self.a*np.ones([line_num])
        
    
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

    def line_of_sight(self,
                      obsv_node,
                      tgt_node,
                      ext_poly):
        visibility = np.ones([obsv_node.shape[0],1])
        inter_set = []
        outer_set = []
        for i in range(obsv_node.shape[0]):
            inter = shapely.intersection(ext_poly,LineString([obsv_node[i],tgt_node[i]]))
            outer = LineString([obsv_node[i],tgt_node[i]])-inter
            
            inter_set.append(inter)
            outer_set.append(outer)
            
            if len(outer.xy[0])!=0:
                visibility[i] = 0
        
        return visibility,inter_set,outer_set
        


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
        p = np.asarray(p,dtype=np.float32).reshape([-1,2])
        Ps = np.asarray(Ps,dtype=np.float32).reshape([-1,2])
        return np.sqrt(np.sum((p-Ps)**2,1))
        
    @staticmethod
    def midP(P1,P2):
        return (P1+P2)/2
    
    @staticmethod
    def totalLength(Ps):
        diff_dist_arr = diff_dist(Ps)
        return np.sum(diff_dist_arr)

    @staticmethod
    def diff_dist(Ps):
        return np.sqrt(np.sum(np.diff(Ps,axis=0)**2,axis=1))
    
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
    