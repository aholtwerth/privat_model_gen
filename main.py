'Main file for model generation.'
from Hinging_Hyperplanes.HHTA import Linear_model
import numpy as np

Linear_model
test0 = (np.random.rand(200,2)-0.5)*2
test1 = np.zeros((200,1))
test1[:,0] = np.exp(test0[:,0])
data = list()
data.append(test0)
data.append(test1)
data[0][3,0] = np.nan
data[0][14,1] = np.nan
data[1][2,0] = np.nan

LM = Linear_model(data)

LM.add_linear_element(LM.data)


