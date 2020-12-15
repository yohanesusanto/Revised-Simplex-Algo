import numpy as np
from numpy.linalg import norm, inv

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
