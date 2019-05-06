from mxnet import ndarray as F


feature = F.arange(2*6*7*7).reshape(2, 6, 7, 7)
print(feature.shape)
print(feature.reshape(0, -4, 2, -1, -2).shape)
print(feature.reshape(0, -4, 2, -1, -2).swapaxes(1, 2).shape)
print(feature.reshape(0, -4, 2, -1, -2).swapaxes(1, 2).reshape(0, -3, -2).shape)




