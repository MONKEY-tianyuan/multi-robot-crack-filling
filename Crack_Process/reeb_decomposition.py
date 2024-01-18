import numpy as np
import copy as cp
from utils import *
import cv2
import rasterio.features
from matplotlib import pyplot as plt
from shapely.ops import linemerge, unary_union, polygonize
from shapely import Polygon,LineString,MultiLineString
from itertools import combinations


class Reeb_Decomp():
    def __init__(self,
                 scanSpace = None,
                 objCrack = None) -> None:
        super(Reeb_Decomp,self).__init__()
        self.scanSpace = scanSpace
        self.objCrack = objCrack
        self.W = self.scanSpace.bounds[2]
        self.H = self.scanSpace.bounds[3]
        
        #critical point sets (V)
        self.critPt = []
        self.critPt_np = None
        self.critPt_sorted_np = None
        self.critPt_red = None
        self.critPt_red_comp = None
        self.critPt_final = []
        self.critPt_del = []
        
        #split line sets (E)
        self.splitLine = None
            #devide splineLine set into mergeLine and nonemergeLine
        self.mergeLine = []
        self.nonmergeLine = None
        
        self.splitEdge = None
        
        #cell sets (C)
        self.Cell = []
        self.Cell_shrunk = []
        self.Cell_red = []
        self.cell_Visibility = []
        self.edgeList = []
        
        #Reeb representation I
        self.critpt_Visibility = None
    
    def morse_decomposition(self):
        width = int(self.W)
        height = int(self.H)
        
        #get boundary of complete coverage space
        boundary = extract_boundary(self.scanSpace)
        
        boundary_dense = boundary_interpolate(boundary=boundary,interpolate_dist=50)
        
        #find critical points of the space
        for part_boundary in boundary:
            subcritP = find_critical_point(part_boundary)
            self.critPt.append(subcritP)
        
        #critical point list to array
        critPt_np = self.critPt[0]
        for i in range(1,len(self.critPt)):
            critPt_np = np.concatenate([critPt_np,self.critPt[i]],axis=0)
        self.critPt_np = critPt_np
        #sort critical points: left to right
        self.critPt_sorted_np = critPt_np[np.argsort(critPt_np[:,0])]
        #copy to reduced critical point set (initialization)
        self.critPt_red = cp.deepcopy(self.critPt_sorted_np)
        #obtain split lines based on sorted critical points
        self.split_line()
        #decompose coverage space into cells by split lines
        self.decompose_to_cell()
        
        #merge small-area cells with larger ones
        self.Cell_red = cp.deepcopy(self.Cell)
        cell_area_arr = [cell.area for cell in self.Cell_red]
        self.cell_reduction(cell_area_arr,red_threshold=6000)
        #cell visibility(for Reeb Graph representation II)
        self.cell_Visibility = self.cell_visibility(self.Cell_red)
        self.compensate_critical_point()
        self.critical_point_visibility()
        #Reeb Graph representation I
        
        #delete merged-area critical points
        #self.filter_critical_point()
        #visualization
        # for cell in self.Cell_red:
        #     gd.GeoSeries(cell).plot()
        print(1)
        return self.critPt_final, self.edgeList, self.Cell_red
    
    def critical_point_visibility(self):
        num_critPt = self.critPt_final.shape[0]
        #num_critPt_del = self.critPt_del.shape[0]
        self.critpt_Visibility = np.zeros([num_critPt,num_critPt])
        for cell in self.Cell_red:
            visible_set = []
            for i in range(num_critPt):
                if shapely.intersects(cell,Point(self.critPt_final[i])):
                    visible_set.append(i)
            combins = [c for c in combinations(visible_set, 2)]
            for i,j in combins:
                self.critpt_Visibility[i,j] = self.critpt_Visibility[j,i] = True
        
        #merge the reduced critical-point visible neighbour connection into finial critical point visibility matrix        
        for point in self.critPt_del:
            ind = np.argmin(spdist(point,self.critPt_final))
            #look for visible points for point
            vis = np.zeros([num_critPt])
            for cell in self.Cell_red:
                if shapely.intersects(cell,Point(point)):
                    for i in range(num_critPt):
                        if i!=ind:
                            if shapely.intersects(cell,Point(self.critPt_final[i])):
                                vis[i] = True
            self.critpt_Visibility[ind,:] += vis
            self.critpt_Visibility[:,ind] += vis

        
        self.critpt_Visibility = np.where(self.critpt_Visibility,1,0)
        
        # for i in range(num_critPt):
        #     for j in range(i+1,num_critPt):
        edge_start,edge_end = np.where(np.triu(self.critpt_Visibility))  
        self.edgeList = np.concatenate([edge_start.reshape([-1,1]),edge_end.reshape([-1,1])],axis=1)     
        
    
    def compensate_critical_point(self):
        self.critPt_red_comp = cp.deepcopy(self.critPt_red)
        for cell in self.Cell_red:
            count = 0
            for point in self.critPt_red_comp:
                if shapely.intersects(cell,Point(point)):
                    count += 1
                    point_tmp = point
            if count < 2:
                for point2 in self.critPt_sorted_np:
                    if Point(point2) != Point(point_tmp) and shapely.intersects(cell,Point(point2)):
                        self.critPt_red_comp = np.concatenate([self.critPt_red_comp,point2.reshape([-1,2])],axis=0)
                        break
        
        self.critPt_final.append(self.critPt_red_comp[0])
        
        
        #reduce critical points by distance
        
        for point in self.critPt_red_comp:
            if np.min(spdist(point,np.array(self.critPt_final)))>25:
                self.critPt_final.append(point)
            else:
                self.critPt_del.append([point])
        self.critPt_final = np.array(self.critPt_final)
        self.critPt_del = np.array(self.critPt_del)
                    
    def filter_critical_point(self):
        for crit_pt in self.critPt_sorted_np:
            nonmerge_flag = 0
            merge_flag = 0
            for line in self.nonmergeLine:
                if shapely.intersects(line,Point(crit_pt)):
                    nonmerge_flag = 1
            
            # if not nonmerge_flag:
            #     for line in self.mergeLine:
            #         if shapely.intersects(line,Point(crit_pt)):
            #             merge_flag = 1
            
            if not nonmerge_flag:
                self.critPt_red = np.delete(self.critPt_red,point_loc_in_set(crit_pt,self.critPt_red)[0],axis=0)
                
    def cell_reduction(self,cell_area_arr,red_threshold):
        while min(cell_area_arr)<red_threshold:
            cell_area_arr = self.cell_reduction_merge(cell_area_arr)
            
    def cell_reduction_merge(self,cell_area_arr):
        #Cell_red_cp = cp.deepcopy(self.Cell_red)
        unfinish_flag = 1
        
        while unfinish_flag:
            unfinish_flag = 0
            visibility_mat = self.cell_visibility(self.Cell_red)
            Cell_ind_red = []
            Cell_red = []
            
            for ind,cell_area in enumerate(cell_area_arr):
                if cell_area > 5000:
                    Cell_ind_red.append(ind)
                    Cell_red.append(self.Cell_red[ind])
            
            for ind,cell_area in enumerate(cell_area_arr):
                if cell_area <= 5000:
                    visibility = visibility_mat[ind]
                    merge_ind = np.where(visibility)[0][0]
                    #delete merging critical point
                    merging_line = shapely.intersection(self.Cell_red[ind],self.Cell_red[merge_ind])
                    for ind2,critpt in enumerate(np.flip(self.critPt_red,axis=0)):
                        if shapely.intersects(Point(critpt),merging_line):
                            del_ind = self.critPt_red.shape[0]-1-ind2
                            self.critPt_red = np.delete(self.critPt_red,del_ind,axis=0)
                    
                    # for ind2,line in enumerate(self.nonmergeLine):
                    #     if type(shapely.intersection(merging_line,line)) is LineString:
                    #         if line_to_numpy(shapely.intersection(merging_line,line)).size>0:
                    #             nonmerge_line = line-merging_line
                    #             nonmerge_line = gd.GeoSeries(nonmerge_line).explode()[0]
                    #             for line in nonmerge_line:
                    #                 if line.length>2:
                    #                     nonmerge_line2 = line
                    #                     break
                    #             self.nonmergeLine[ind2] = nonmerge_line2
                    #             self.mergeLine.append(line)
                    #merge polygons
                    merge_poly = shapely.union(self.Cell_red[ind],self.Cell_red[merge_ind])
                    
                    switch_ind = np.where(np.array(Cell_ind_red)==merge_ind)[0]
                    
                    if switch_ind.size == 0:
                        unfinish_flag = 1
                        Cell_ind_red.append(ind)
                        Cell_red.append(self.Cell_red[ind])
                    else:
                        Cell_red[switch_ind[0]] = merge_poly
            
            self.Cell_red = Cell_red
            cell_area_arr = [cell.area for cell in self.Cell_red]
        return cell_area_arr
    
    # def cell_reduction(self,cell_area_arr):
    #     while min(cell_area_arr)<5000:
    #         visibility_mat = self.cell_visibility(self.Cell_red)
    #         cell_area_arr = self.cell_reduction_merge(visibility_mat,cell_area_arr)
    #         print(1)
    
    # def cell_reduction_merge(self,visibility_mat,cell_area_arr):
    #     #cell_area_arr = [cell.area for cell in self.Cell_red]
    #     for ind,cell_area in enumerate(cell_area_arr):
    #         if cell_area < 5000:
    #             visibility = visibility_mat[ind]
    #             merge_ind = np.where(visibility)[0][0]
    #             merge_poly = shapely.union(self.Cell_red[ind],self.Cell_red[merge_ind])
    #             self.Cell_red[ind] = self.Cell_red[merge_ind] = merge_poly
                
    #             print(1)
    #     self.Cell_red = get_unique_element(self.Cell_red)
    #     cell_area_arr = [cell.area for cell in self.Cell_red]
    #     return cell_area_arr

                
    
    @staticmethod
    def cell_visibility(cell_list):
        visibility_mat = np.zeros([len(cell_list),len(cell_list)])
        for i in range(len(cell_list)-1):
            for j in range(i+1,len(cell_list)):
                 visibility_mat[i,j]= shapely.intersects(cell_list[i],cell_list[j]) and (type(shapely.intersection(cell_list[i],cell_list[j])) is LineString) 
        visibility_mat = np.triu(visibility_mat) + np.triu(visibility_mat, 1).T
        return visibility_mat    
    
    def cell_shrink_for_visibility(self):
        for cell in self.Cell:
            cell_shrunk = shrink_or_swell_shapely_polygon(cell,factor=0.001)
            self.Cell_shrunk.append(cell_shrunk)
           
    def decompose_to_cell(self):
        scanSpace = cp.deepcopy(self.scanSpace)
        scanSpace = gd.GeoSeries(scanSpace).explode()[0]
        for line in self.splitLine:
            remain = Polygon([])
            for cell in scanSpace:
                if not shapely.intersects(cell,line):
                    self.Cell.append(cell)
                else:
                    remain = remain.union(cell)
            scanSpace = remain
            scanSpace = shapely.ops.split(scanSpace,line)
            # if scanSpace.area != 0:
            scanSpace = gd.GeoSeries(scanSpace).explode()[0]
            # else:
            #     break
        # if scanSpace.area != 0:
        self.Cell.append(scanSpace[0])


    def cut_polygon_by_line(polygon, line):
        merged = linemerge([polygon.boundary, line])
        borders = unary_union(merged)
        polygons = polygonize(borders)
        return list(polygons)
    
    def split_line(self):
        split_line = []
        for crit_pt in self.critPt_sorted_np:
            crit_pt_x = crit_pt[0]
            line = LineString([[crit_pt_x,0],[crit_pt_x,self.H]])
            line_intersect = shapely.intersection(self.scanSpace,line)
            if type(line_intersect) is LineString:
                split_line.append(line_intersect)
            elif type(line_intersect) is MultiLineString:
                line_tmp = []
                line_intersect = gd.GeoSeries(line_intersect).explode()[0]
                for line in line_intersect:
                    if shapely.intersects(Point(crit_pt),line):
                        line_tmp.append(line)
                if len(line_tmp)>1:
                    line_tmp_np = line_to_numpy(line_tmp[0])
                    for i in range(1,len(line_tmp)):
                        line_tmp_np = np.concatenate([line_tmp_np,line_to_numpy(line_tmp[i])],axis=0)
                    # make sure the line can split the polygon successfully
                    y_min = np.min(line_tmp_np[:,1])-1
                    y_max = np.max(line_tmp_np[:,1])+1
                    line_tmp = LineString([[crit_pt_x,y_min],[crit_pt_x,y_max]])
                elif len(line_tmp)==1:
                    # make sure the line can split the polygon successfully
                    line_tmp_np = line_to_numpy(line_tmp[0])
                    y_min = np.min(line_tmp_np[:,1])-1
                    y_max = np.max(line_tmp_np[:,1])+1
                    line_tmp = LineString([[crit_pt_x,y_min],[crit_pt_x,y_max]])
                split_line.append(line_tmp)
        self.splitLine = cp.deepcopy(split_line)
        self.nonmergeLine = cp.deepcopy(split_line)
        
    
        
        
#general functions
def detect_local_minima(arr):
    
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)   

def find_critical_point(boundary):
    #find x axis critical points
    grad_x_arr = np.diff(boundary[:,0])
    cp = []
    diff_len = grad_x_arr.shape[0]
    for i in range(diff_len):
        if grad_x_arr[i] == 0:
            cp.append((boundary[i]+boundary[i+1])/2)
    
    for i in range(1,diff_len-1):
        if grad_x_arr[i]!=0 and (grad_x_arr[i-1]<=0)!=(grad_x_arr[i]<0):
            cp.append(boundary[i])
    cp = np.array(cp)
    return cp

def boundary_interpolate(boundary,
                         interpolate_dist=20):
    boundary_dense = []
    for part_boundary in boundary:
        part_boundary_dense = cp.deepcopy(part_boundary)
        dist_arr = diff_dist(part_boundary)
        for ind,dist in enumerate(dist_arr):
            if dist>interpolate_dist:
                start = part_boundary[ind]
                end = part_boundary[ind+1]
                interpolate_num = int(dist//interpolate_dist)
                diff = (end-start)/(interpolate_num+1)
                interpolate_arr = np.zeros([interpolate_num,2])
                for i in range(interpolate_num):
                    interpolate_arr[i] = start + (i+1)*diff
                    part_boundary_dense = np.insert(part_boundary_dense,ind+1,np.array([0.0,0]),axis=0)
                part_boundary_dense[ind+1:(ind+interpolate_num+1)] = interpolate_arr
        boundary_dense.append(part_boundary_dense)
    return boundary_dense

def extract_boundary(poly):
        boundary = poly.boundary
        if type(boundary) is MultiLineString:
            boundary_np = []
            boundary = gd.GeoSeries(boundary).explode()[0]
            for part_boundary in boundary:
                part_boundary = part_boundary.xy
                boundary_np.append(boundary_to_numpy(part_boundary))
            
        elif type(boundary) is LineString:
            boundary = boundary.xy
            boundary_np = boundary_to_numpy(boundary)
            
        return boundary_np

def boundary_to_numpy(boundary):
    return np.concatenate([np.asarray(boundary[0]).reshape([-1,1]),np.asarray(boundary[1]).reshape([-1,1])],axis=1)

def line_to_numpy(line):
    line = line.xy
    return np.asarray(line).T