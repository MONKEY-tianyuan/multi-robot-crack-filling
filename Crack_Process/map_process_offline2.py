import os
from matplotlib import pyplot as plt
from crack_graph import Crack_Graph
from reeb_decomposition import Reeb_Decomp
intput_dir = 'target_generation/Data/CrackMaps'
output_dir = 'target_generation/Data/CrackMaps_MR'

image_path = os.path.join(intput_dir,'Uniform','myCrack6_90_2.png')

#crack preprocessing
crack_model = Crack_Graph(image_path)
crack_nodeList,crack_edgeList,crack_edgeIndList,objCrack,scanSpace = crack_model.run()

#morse decomposition and reeb decomposition
reeb_decomp = Reeb_Decomp(scanSpace=scanSpace,objCrack = objCrack)
critPt, reeb_edgeList, reeb_Cell=reeb_decomp.morse_decomposition()

plt.figure(num=1)
for partial_crack in crack_model.Partial_Crack_list:
    plt.plot(partial_crack[:,0],partial_crack[:,1])


plt.show()
