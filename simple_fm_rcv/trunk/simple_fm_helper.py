a_glob=0
b_glob=0
import numpy
def freq_select(a,b):
    global a_glob
    global b_glob
    
    if a != a_glob:
        a_glob = a
        return a
        
    elif b != b_glob:
        b_glob = b
        return b
    else:
        a_glob = a
        return a
    
