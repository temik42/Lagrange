import numpy as np
import sys
sys.path.append('Q:\\python\\lib')
from utils import put, take

    
def boundary(a, dim, types):
    id1 = [0,-1]
    id2 = [-2, 1]
    
    for i in id1:
        type = types[i]
    
        if (type == 'zero'):
            put(a, 0, i, dim)
        else:

            if (type == 'uniform'):
                ii = id2[i]
                inv = 1

            if (type == 'continuous'):
                ii = id2[i-1]
                inv = 1

            if (type == 'mirror'):
                ii = id2[i-1]
                inv = -1

            put(a, inv*take(a, ii, dim), i, dim)
        

        