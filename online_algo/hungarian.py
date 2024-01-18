import numpy as np 
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
 

cost_matrix = np.array([
    [15,40,45,20,60,1],
    [20,60,35,12,33,40],
    [20,40,25,45,66,5]
])


matches = linear_sum_assignment(cost_matrix)
print('scipy API result:\n', matches)