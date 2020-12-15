import numpy as np
from numpy.linalg import norm, inv
from simplex_step import simplex_step

def simplex_step(A,b,c,iB,iN,xB,Binv,irule):
  coefficient_base = []
  for i in (iB):
    coefficient_base.append(c[0,i-1]) #getting CB^T(coefficient of the basic variables)
  w = np.dot(coefficient_base,Binv) #getting w
  z = np.matrix(np.copy(c))
  for i in iN:
    z[0,i-1] = np.dot(w,A[:,i-1])

  if (irule == 0): #irule = 0 indicates that the smallest coefficient rule should be used
    reduced_cost = 0
    for i in iN: #finding the negative reduced cost from the reduced cost list of the non-basic variables
      if c[0,i-1]-z[0,i-1]<reduced_cost:
        reduced_cost = c[0,i-1]-z[0,i-1]
        entering_variable = i
    if reduced_cost == 0: #if there is none, we are at optimality
      istatus = -1 #at optimality
      return [istatus,iB,iN,xB,Binv]

    y = np.dot(Binv,A[:,entering_variable-1])
    current_ratio = float('inf') #negative constant
    for i in range(y.size): #finding the minimum ratio
      if y[i,:] > 0:
        if (xB[i,:]/y[i,:] >= 0) and (xB[i,:]/y[i,:] < current_ratio):
          current_ratio = xB[i]/y[i,:]
          leaving_index = i

    if (current_ratio ==  float('inf')): #if there is no ratio >= 0, we return the program is unbounded
      print("Program is unbounded")
      istatus = 16 #program is unbounded
      return [istatus,iB,iN,xB,Binv]

  elif (irule == 1):
    sorted_iN = iN
    sorted_iN.sort()
    reduced_cost = 0
    for i in sorted_iN: #finding the negative reduced cost from the reduced cost list of the non-basic variables
      if c[0,i-1]-z[0,i-1]<0:
        reduced_cost = c[0,i-1]-z[0,i-1]
        entering_variable = i
        break
    if reduced_cost == 0: #if there is none, we are at optimality
      istatus = -1 #at optimality
      return [istatus,iB,iN,xB,Binv]
    y = np.dot(Binv,A[:,entering_variable-1])

    current_ratio = float('inf') #negative constant
    for i in range(y.size): #finding the minimum ratio
      if y[i,:] > 0:
        if (xB[i,:]/y[i,:] >= 0) and (xB[i,:]/y[i,:] < current_ratio):
          current_ratio = xB[i]/y[i,:]
          leaving_index = i
          current_variable = iB[i]
        elif (xB[i,:]/y[i,:] >= 0) and (xB[i,:]/y[i,:] == current_ratio):
          if current_variable < iB[i]:
            current_ratio = xB[i]/y[i,:]
            leaving_index = i
            current_variable = iB[i]

    if (current_ratio ==  float('inf')): #if there is no ratio >= 0, we return the program is unbounded
      istatus = 16 #program is unbounded
      print("Program is unbounded")
      return [istatus,iB,iN,xB,Binv]

  entering_index = iN.index(entering_variable)
  leaving_variable = iB[leaving_index]
  print("leaving index = " + str(leaving_index))
  print("entering index = " + str(entering_index))
  print("leaving variable = " + str(leaving_variable))
  print("entering variable = " + str(entering_variable))
  iB.remove(leaving_variable) #removing the leaving variable from iB
  iB.insert(leaving_index,entering_variable) #inserting the entering variable to iB
  iN.remove(entering_variable) #removing the entering variable from iN
  iN.insert(entering_index,leaving_variable) #inserting the leaving variable to iN


  eta = np.matrix(np.copy(b))
  for i in range(len(A[:])):
    if (i == leaving_index):
      eta[i,0] = 1/A[leaving_index,entering_variable-1]
    else:
      eta[i,0] = -1*A[i,entering_variable-1]/A[leaving_index,entering_variable-1]
  E=np.matrix(np.eye(len(iB)))
  E[:,leaving_index] = eta
  if (np.linalg.det(A[:,[index_B-1 for index_B in iB]]) == 0):
    istatus = 16 #infeasible
    return istatus,iB,iN,xB,A[:,[index_B-1 for index_B in iB]]
  else:
    Binv = A[:,[index_B-1 for index_B in iB]].I #getting Binv from A
  
  xB = np.dot(Binv,b)

  istatus = 0
  return [istatus,iB,iN,xB,Binv]

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
