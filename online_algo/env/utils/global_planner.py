import torch
import torch.nn as nn

class GlobalPlanner(nn.Module):
    def __init__(self,
                 Robot=None,
                 Decom_Cell=None,
                 Decom_Node = None,
                 Decom_Edge = None,
                 Distance = None) -> None:
        super(GlobalPlanner,self).__init__()
        self.Robot = Robot
        self.Decom_Cell = Decom_Cell
        self.Distance = Distance
        self.Decom_Node = Decom_Node
        self.Decom_Edge = Decom_Edge
        
    
        

        

            