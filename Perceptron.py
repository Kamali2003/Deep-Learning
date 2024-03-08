#OR GATE

import numpy as np
x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
y = np.array([0,1,1,1])
w1 = 1
w2 = 1
theta = 1                                                         #threshold
f = x1*w1+x2*w2
y_pred = (f>=theta).astype(int)
if np.all(y == y_pred):
         print(f"f = {f}\ny_pred = {y_pred}\ncorrect weights and threshold")
         print("w1=",w1,"w2=",w2)
         print("threshold=",theta)
else:
         print("change the weights/threshold")

#AND GATE

import numpy as np
x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
y = np.array([0,0,0,1])
w1 = 1
w2 = 1
theta = 2                                                              #threshold
f = x1*w1+x2*w2
y_pred = (f>=theta).astype(int)
if np.all(y == y_pred):
  print(f"f = {f}\ny_pred = {y_pred}\ncorrect weights and threshold")
  print("w1=",w1,"w2=",w2)
  print("threshold=",theta)
else:
  print("change the weights/threshold")

#NOR GATE

import numpy as np
x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
y = np.array([1,0,0,0])
w1 = w2 = 1
theta = 0                                                         #threshold
f = x1*w1+x2*w2
y_pred = (f>=theta).astype(int)
if np.all(y == y_pred):
  print(f"f = {f}\ny_pred = {y_pred}\ncorrect weights and threshold")
  print("w1=",w1,"w2=",w2)
  print("threshold=",theta)
else:
  print("change the weights/threshold")

#NAND GATE
import numpy as np
x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
y = np.array([1,1,1,0])
w1 = w2 = 1
theta = 1                                                        #threshold
f = x1*w1+x2*w2
y_pred = (f>=theta).astype(int)
if np.all(y == y_pred):
  print(f"f = {f}\ny_pred = {y_pred}\ncorrect weights and threshold")
  print("w1=",w1,"w2=",w2)
  print("threshold=",theta)
else:
  print("change the weights/threshold")
