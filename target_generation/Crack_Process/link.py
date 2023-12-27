import numpy as np

class Link():
    def __init__(self,
                 branch_len = None) -> None:
        super(Link,self).__init__()

        self.x = []
        self.y = []
        for i in range(branch_len):
            self.x.append([])
            self.y.append([])
        