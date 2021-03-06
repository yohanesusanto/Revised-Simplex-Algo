import numpy as np
from numpy.linalg import norm, inv
from simplex_step import simplex_step #for the extra credit file
#from simplex_step1 import simplex_step #for the non-extra credit file

def simplex_init(A,b,c):
  iB = [] #list of variables that are in the basis
  iN = [] #list of variables that are not in the basis
  for i in range (c.size):
    iN.append(i+1) #initial setup for the nonbasic variables
  for i in range (len(A[:])):
    iB.append(c.size + i + 1) #initial setup for the basic variables
  for i in range(b.size):
    if (b[i,0] < 0):
      b[i,0] = -1*b[i,0]
      A[i] = -1*A[i]
  xB = np.matrix(np.copy(b)) #initial xB is equal to b
  A = np.hstack((A,np.eye(len(A[:])))) #adding artifical variables to matrix A
  constant_reduced_cost = np.hstack((np.zeros(c.size),np.ones(len(A[:])))) #finding the initial list of the reduced cost
  dynamic_reduced_cost = np.matrix(np.copy(constant_reduced_cost))
  for i in range (len(A[:])): #zeroing out the reduced cost of the artificial variables
    dynamic_reduced_cost = dynamic_reduced_cost - A[i]
  if (np.linalg.det(A[:,[index_B-1 for index_B in iB]]) == 0): #checking if B is invertible
    istatus = 16 #infeasible
    return istatus,iB,iN,xB
  else:
    Binv = A[:,[index_B-1 for index_B in iB]].I #getting Binv from A
  coefficient_base = []
  for i in (iB):
    coefficient_base.append(constant_reduced_cost[i-1]) #getting CB^T(coefficient of the variables in the basis)
  done = 0
  optimize = 0
  while optimize == 0: #letting the program run until done is changed
    [optimize,iB,iN,xB,Binv] = simplex_step(A,b,dynamic_reduced_cost,iB,iN,xB,Binv,0)
    coefficient_base = [] #getting CB^T(coefficient of the variables in the basis)
    for i in (iB):
      coefficient_base.append(constant_reduced_cost[i-1]) #getting CB^T(coefficient of the variables in the new basis)
    for i in range(dynamic_reduced_cost.size):
      dynamic_reduced_cost[0,i] = constant_reduced_cost[i] - np.dot(np.dot(coefficient_base,Binv),A[:,i])
  
  for i in range(dynamic_reduced_cost.size,c.size,-1): #checking all artificial variables
    for j in range(len(iB)): #number of variables in the basis
      if iB[j] == i: #if there is still artifical variables in the basis
        if (xB[j,0] != 0): #if the artificial variables have values that are not 0
          istatus = 16 #infeasible
          return istatus,iB,iN,xB
        else: #if the artificial variables have values that are 0
          istatus = 4 #initialization procedure failed
          return istatus,iB,iN,xB
  istatus = 0 #we have found an optimal vector
  return istatus,iB,iN,xB
