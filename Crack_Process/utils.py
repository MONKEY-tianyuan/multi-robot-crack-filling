import numpy as np
import shapely
import geopandas as gd
from shapely import buffer,Point,LineString,MultiPolygon
from shapely.geometry.multilinestring import MultiLineString
from itertools import combinations
import networkx as nx
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import copy as cp

class Link():
    def __init__(self,
                 branch_len = None) -> None:
        super(Link,self).__init__()

        self.x = []
        self.y = []
        for i in range(branch_len):
            self.x.append([])
            self.y.append([])


class GeoBlock():
    def __init__(self,
                 Poly =None,
                 idx = None) -> None:
        super(GeoBlock,self).__init__()
        
        self.Poly = Poly
        self.idx = idx
        
class Vertice():
    def __init__(self,
                 inpoly_ind = None,
                 node_pos = None,
                 node_set = np.array([])) -> None:
        super(Vertice,self).__init__()
        self.inpoly_ind = inpoly_ind
        
        if node_set.size==0:
            self.node_pos = node_pos
        else:
            self.node_pos = node_set[inpoly_ind]

class Visbility():
    def __init__(self,
                 visibility=None,
                 inter = None,
                 outer = None,) -> None:
        super(Visbility,self).__init__()
        self.visibility = visibility
        self.inter = inter
        self.outer = outer


def shrink_or_swell_shapely_polygon(my_polygon, factor=0.10, swell=False):
    ''' returns the shapely polygon which is smaller or bigger by passed factor.
        If swell = True , then it returns bigger polygon, else smaller '''
    from shapely import geometry

    #my_polygon = mask2poly['geometry'][120]

    shrink_factor = factor 
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*shrink_factor

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance) #expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance) #shrink

    # #visualize for debugging
    # x, y = my_polygon.exterior.xy
    # plt.plot(x,y)
    # x, y = my_polygon_resized.exterior.xy
    # plt.plot(x,y)
    # # to net let the image be distorted along the axis
    # plt.axis('equal')
    # plt.show()    
    
    return my_polygon_resized
        
def get_unique_element(list):
    unique = []
    for element in list:
        if element not in unique:
            unique.append(element)
    return unique        
        
        
        
def pathfinder(start,goal, poly):
    start = start.reshape([-1,2])
    goal = goal.reshape([-1,2])
    ext_boundary = poly.boundary.xy
    ext_boundary = np.concatenate([np.asarray(ext_boundary[0]).reshape([-1,1]),np.asarray(ext_boundary[1]).reshape([-1,1])],axis=1)
    
    init_comb_nodes = np.concatenate([start,ext_boundary,goal],axis=0)
    dist = np.inf*np.ones([init_comb_nodes.shape[0],1])
    init_comb_nodes = np.concatenate([init_comb_nodes,dist],axis=1)    

    ii = np.array([c for c in combinations(range(init_comb_nodes.shape[0]),2)])
    dist = spdist(init_comb_nodes[ii[:,0],:2],init_comb_nodes[ii[:,1],:2]).reshape([-1,1])
    
    visibility_h,_,_ = line_of_sight2(init_comb_nodes[ii[:,0],:2],init_comb_nodes[ii[:,1],:2],poly)
    
    ii = np.concatenate([ii,dist],axis=1)
    
    visible_edge = ii[np.where(visibility_h)[0]]
    
    G = nx.Graph()
    for i in range(init_comb_nodes.shape[0]):
        G.add_node(i)
    G.add_weighted_edges_from(visible_edge)
    
    p = nx.shortest_path(G,source=0,target=init_comb_nodes.shape[0]-1,weight='weight')
    d = nx.shortest_path_length(G,source=0,target=init_comb_nodes.shape[0]-1,weight='weight')
    waypoint = np.array([init_comb_nodes[int(i),:2] for i in p])
    return waypoint,d
    
          
        
def line_of_sight2(obsv_node,
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
        
        if type(outer) is MultiLineString:
            visibility[i] = 0
        elif type(outer) is LineString:
            if len(outer.xy[0])!=0:
                visibility[i] = 0
    
    return visibility,inter_set,outer_set        
        
        
        

def conv2D_from_block(conv_block):
    img_conv = conv_block
    kernel = np.array([[1,1,1],
                    [1,0,1],
                    [1,1,1]])
    
    return np.sum(np.multiply(img_conv,kernel))


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
    
    
def spdist(p,Ps):
    p = np.asarray(p,dtype=np.float32).reshape([-1,2])
    Ps = np.asarray(Ps,dtype=np.float32).reshape([-1,2])
    return np.sqrt(np.sum((p-Ps)**2,1))
    

def midP(P1,P2):
    return (P1+P2)/2


def totalLength(Ps):
    diff_dist_arr = diff_dist(Ps)
    return np.sum(diff_dist_arr)

def diff_dist(Ps):
    return np.sqrt(np.sum(np.diff(Ps,axis=0)**2,axis=1))

def inpxMap(x):
    return int(x*25.4/2)


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

def point_loc_in_set(pt,P):
    return np.where((P==pt).all(axis=1))[0]