import numpy as np
from numpy.linalg import norm, inv
from simplex_init import simplex_init
from simplex_step import simplex_step

def simplex_method(A,b,c,irule):
  [init,iB,iN,xB] = simplex_init(A,b,c)
  if init == 4:
    X = np.zeros((6,1),dtype = np.float64)
    eta = 0
    istatus = 16
    return istatus,X,eta,iB,iN,xB
  
  elif init == 16:
    X = np.zeros((6,1),dtype = np.float64)
    eta = 0
    istatus = 4
    return istatus,X,eta,iB,iN,xB
  
  else:
    for i in range(c.size+len(iB),c.size,-1): #removing artificial variables from the list of non-basic variables
      iN.remove(i)
    if (np.linalg.det(A[:,[index_B-1 for index_B in iB]]) == 0): #checking if B is invertible
      istatus = 16 #infeasible
      return istatus,iB,iN,xB
    else:
      Binv = A[:,[index_B-1 for index_B in iB]].I #getting Binv from A
    xB = np.dot(Binv,b)
    one_step = 0
    while one_step == 0:
      [one_step,iB,iN,xB,Binv] = simplex_step(A,b,c,iB,iN,xB,Binv,irule)
      
    if one_step == 16:
     istatus = 32
     X = np.zeros((6,1),dtype = np.float64)
     X[[(b1-1) for b1 in iB]] = xB
     eta = np.dot(c,X)
     return istatus,X,eta,iB,iN,xB

    else:
      istatus = 0
      X = np.zeros((6,1),dtype = np.float64)
      X[[(b1-1) for b1 in iB]] = xB
      eta = np.dot(c,X)
      return istatus,X,eta,iB,iN,xB
