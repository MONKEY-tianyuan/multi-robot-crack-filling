import torch
import numpy as np
from matplotlib import pyplot as plt

from shapely import buffer,Point,LineString,MultiPolygon
import geopandas as gd

a = torch.tensor([0.2,1.3])
b = torch.tensor([[0,0.0],[1.3,9],[2.3,4]])
print(torch.sqrt(torch.sum((a-b)**2,1)))
print(torch.sqrt(torch.sum(torch.diff(b,dim=0)**2)))

a = np.array([[1,0,1],
              [1,1,1],
              [1,1,1]])
kernel = np.array([[1,1,1],
                    [1,0,1],
                    [1,1,1]])

print(np.sum(np.multiply(a,kernel)))

aRan = buffer(Point(10,10),2)
s = gd.GeoSeries(aRan)
s.plot()
plt.show()