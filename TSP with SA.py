import numpy as np
import random as rand
import pandas as pd
#import matplotlib.pyplot as plt
import time
import math

#%%
# define timer
def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
        
#%%
# import data
#df = pd.DataFrame()
df = pd.read_excel (r'F:\Work\3. Research\0. Code\4. TSP with SA - Python\instance1.xls', sheet_name='Sheet1', header=None)
data = df.as_matrix()
df = pd.read_excel (r'F:\Work\3. Research\0. Code\4. TSP with SA - Python\instance1.xls', sheet_name='Sheet2', header=None)
distanceMatrix = df.as_matrix()
df = pd.read_excel (r'F:\Work\3. Research\0. Code\4. TSP with SA - Python\instance1.xls', sheet_name='Sheet5', header=None)
travelTimeMatrix = df.as_matrix()
del df

#%%
numNode = len(data[:,1])
demand = data[:,3]
#ServiceTime = Data(:,5);
#ServiceTime = zeros(NumNode,1);


#%% define parameters for the heuristic
Tmax = 150
Tmin = 0.01
maxNonImprove = 10000
maxIteration = 1000
coolingRate = 0.99

#%% define parameters for the problem
#CapacityGV = 200; % capacitated GV
vehicleCapacity = sum(demand) #incapacitated GV

#%% solution storage for the heuristic
currentSolution = []
X_penalty = []
Y_penalty = []
counter = 0

#%% define some functions for the heuristic
#%% a | function for fitness calculation
def fitnessTSP(solution, distanceMatrix):
    distance = np.zeros([1,1])
    for i in range(0,len(solution[0,:])-1):
        departureNode = int(solution[0][i])
        nextNode = int(solution[0][i+1])
        distance += distanceMatrix[departureNode][nextNode]
        del(departureNode, nextNode)
    objective = distance
    return objective

#%% b | local search function
#% b.1 | swap '
def swapSearch(solution):
    swapNodes = rand.sample(range(1,len(X_sol[0,:])-1),2)
    swap1 = swapNodes[0]
    swap2 = swapNodes[1]
    
    new_solution = np.array(solution)
    temp2 = np.array(new_solution[0][swap1])
    new_solution[0][swap1] = new_solution[0][swap2]
    new_solution[0][swap2] = temp2
    del (swap1,swap2,temp2,swapNodes)
    return new_solution

#% b.2 | insertion
def insertionSearch(solution):
    insertionNodes = rand.sample(range(1,len(X_sol[0,:])-1),2)
    while abs(insertionNodes[0]-insertionNodes[1]) == 1 or abs(insertionNodes[0]-insertionNodes[1]) == 0:
        insertionNodes = rand.sample(range(1,len(X_sol[0,:])-1),2)
    insert1 = insertionNodes[0]
    insert2 = insertionNodes[1] # target
    
    new_solution = np.array(solution)
    if insert1<insert2:
        temp2 = np.array(new_solution)
        new_solution[0][insert2-1] = new_solution[0][insert1]
        if insert1 == (insert2-2):
            new_solution[0][insert1] = temp2[0][insert1+1]
        else:
            new_solution[0][insert1:insert2-1] = np.array(temp2[0][insert1+1:insert2])            
    elif insert1>insert2:
        temp2 = np.array(new_solution)
        new_solution[0][insert1-1] = np.array(new_solution[0][insert2])
        if insert1 == (insert1-2):
            new_solution[0][insert2] = np.array(temp2[0][insert2+1])
        else:
            new_solution[0][insert2:insert1-1] = np.array(temp2[0][insert2+1:insert1])
    del (insertionNodes,insert1,insert2)
    return new_solution

#% b.3 | 2-opt (double bridge)
def twoOptSearch(solution):
    new_solution = np.array(solution)
    edgeNodes = rand.sample(range(1,len(X_sol[0,:])-2),2)
    while abs(edgeNodes[0]-edgeNodes[1]) == 1:
        edgeNodes = rand.sample(range(1,len(X_sol[0,:])-2),2)
    np.sort(edgeNodes)    
    edgeA = edgeNodes[0]
    edgeB = edgeNodes[0]+1
    edgeC = edgeNodes[1]
    edgeD = edgeNodes[1]+1
    
    #swap them with 2-opt rule
    nodeA = new_solution[0][edgeA]
    nodeB = new_solution[0][edgeC]
    nodeC = new_solution[0][edgeB]
    nodeD = new_solution[0][edgeD]
    new_solution[0][edgeA] = nodeA
    new_solution[0][edgeB] = nodeB
    new_solution[0][edgeC] = nodeC
    new_solution[0][edgeD] = nodeD
    del (edgeA, edgeB, edgeC, edgeD, nodeA, nodeB, nodeC, nodeD)
    return new_solution


#%%

#%%
#% simulated annealing main algorithm    
tic()

#% 1 | initialize temperature and iteration
temperature = Tmax
iteration = 1
nonImprove = 0

#% 2 | create initial solution with random solution
X_sol = np.zeros([1,numNode+1], dtype = int)
X_sol[0,1:-1] = np.random.permutation(numNode-1)+1

#% 3 | evaluate the solution
X_objective = fitnessTSP(X_sol, distanceMatrix)

#% 4 | Main iteration of the simulated annealing
while temperature >= Tmin:
    
#%
#% 5 | create new solution based on a random number and the current solution
    r = np.random.random()
    
    if r <= 0.33:
        Y_sol = swapSearch(X_sol)
    elif (r > 0.33) and (r <= 0.66):
        Y_sol = insertionSearch(X_sol)
    else:
        Y_sol = twoOptSearch(X_sol)
    del r
   #%
#% 6 | evaluate the new solution
    Y_objective = fitnessTSP(Y_sol, distanceMatrix)

#%
#% 7 | check if Y_sol is better than X_sol (minimization)
    delta = Y_objective - X_objective
    if delta < 0:
        X_sol = np.array(Y_sol)
        X_objective = np.array(Y_objective)
        nonImprove = 0
    else:
        nonImprove += 1
        r = np.random.random()
        if r < math.exp(-delta/temperature):
            X_sol = np.array(Y_sol)
            X_objective = np.array(Y_objective)
        del r

#%
#% 8 | storing the current solution
    currentSolution.append(X_objective)

#% 9 | checking and update the iteration    
    iteration = iteration+1
    if iteration == maxIteration:
        temperature = temperature*coolingRate
        iteration = 1
        
    if nonImprove >= maxNonImprove:
        break
    
    #print(iteration)
#%
toc()

print(X_objective)