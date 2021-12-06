import numpy as np
keys = ['a', 'b', 'c']

def list2dic(keys):
    values = np.arange(keys.__len__()) + 1
    dic = dict(zip(keys, values))
    return dic

dic1 = list2dic(keys)