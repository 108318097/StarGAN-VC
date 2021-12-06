import numpy as np
keys = ['a', 'b', 'c']
# values = np.arange(keys.__len__()) + 1
values = np.arange(keys.__len__())
dic = dict(zip(keys, values))
print(dic)
