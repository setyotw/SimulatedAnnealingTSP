## Problem: Traveling salesman problem, objective function: minimize total distance traveled
## Solver: Simulated annealing
## Language: Python
## Written by: @setyotw
## Purpose: Public repository
## Date: August 31, 2022

#%% import packages
import numpy as np
import random as rand
import pandas as pd
import time
import math

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS
#%% define timer
# a homemade version of matlab tic and toc functions
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        runtimeCount = time.time() - startTime_for_tictoc
        print ("Elapsed time is " + str(runtimeCount) + " seconds.")
        return runtimeCount
    else:
        print ("Toc: start time not set")
        return int(0)
        
#%% define functions needed
#% a | fitness calculation
def fitnessTSP(solution, distanceMatrix):
    distance = np.zeros([1,1])
    for i in range(0,len(solution[0,:])-1):
        departureNode = int(solution[0][i])
        nextNode = int(solution[0][i+1])
        distance += distanceMatrix[departureNode][nextNode]
        del(departureNode, nextNode)
    objective = distance
    return objective

#% b | neighborhood moves
# b.1 | swap move
def swapSearch(solution):
    # select two different nodes
    swapNodes = rand.sample(range(1,len(solution[0,:])-1),2)
    swap1 = swapNodes[0]
    swap2 = swapNodes[1]
    
    # exchange the position of those nodes
    new_solution = np.array(solution)
    temp2 = np.array(new_solution[0][swap1])
    new_solution[0][swap1] = new_solution[0][swap2]
    new_solution[0][swap2] = temp2
    
    return new_solution

# b.2 | insertion move
def insertionSearch(solution):
    # select two different nodes
    insertionNodes = rand.sample(range(1,len(solution[0,:])-1),2)
    while abs(insertionNodes[0]-insertionNodes[1]) == 1 or abs(insertionNodes[0]-insertionNodes[1]) == 0:
        insertionNodes = rand.sample(range(1,len(solution[0,:])-1),2)
    insert1 = insertionNodes[0]
    insert2 = insertionNodes[1] # target
    
    # move one of the node behind the other one
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
    
    return new_solution

# b.3 | 2-opt (double bridge) move
def twoOptSearch(solution):
    # select four different nodes
    new_solution = np.array(solution)
    edgeNodes = rand.sample(range(1,len(solution[0,:])-2),2)
    while abs(edgeNodes[0]-edgeNodes[1]) == 1:
        edgeNodes = rand.sample(range(1,len(solution[0,:])-2),2)
    np.sort(edgeNodes)    
    edgeA = edgeNodes[0]
    edgeB = edgeNodes[0]+1
    edgeC = edgeNodes[1]
    edgeD = edgeNodes[1]+1
    
    # swap them with 2-opt rule
    nodeA = new_solution[0][edgeA]
    nodeB = new_solution[0][edgeC]
    nodeC = new_solution[0][edgeB]
    nodeD = new_solution[0][edgeD]
    new_solution[0][edgeA] = nodeA
    new_solution[0][edgeB] = nodeB
    new_solution[0][edgeC] = nodeC
    new_solution[0][edgeD] = nodeD
    
    return new_solution

# c | simulated annealing 
def simulatedAnnealing(distanceMatrix, numNodes, temperatureMax, temperatureMin, maxNonImprove, maxIteration, coolingRate):

    # 1 | initialize a solution storage
    solutionRepo = []

    # 2 | initialize a solution storage, current temperature, and iterationCount counters
    temperatureNow = temperatureMax
    iterationCount = int(1)
    nonImprovementCounter = int(0)

    # 3 | create an initial TSP solution with random solution
    currentSol = np.zeros([1,numNodes+1], dtype = int)
    currentSol[0,1:-1] = np.random.permutation(numNodes-1)+1

    # 4 | evaluate the solution and store it as the current & best solution
    currentObjective = fitnessTSP(currentSol, distanceMatrix)
    bestObjective = np.array(currentObjective)
    bestSol = np.array(currentSol)
    
    # 5 | Main iteration of the algorithm
    tic()
    while temperatureNow >= temperatureMin:

        # 5.1 | create a new TSP solution based using a selected neighborhood move
        randomNum = np.random.random()
        if randomNum <= 0.33:
            newSol = swapSearch(currentSol)
        elif (randomNum > 0.33) and (randomNum <= 0.66):
            newSol = insertionSearch(currentSol)
        else:
            newSol = twoOptSearch(currentSol)

        # 5.2 | evaluate the new TSP solution
        newObjective = fitnessTSP(newSol, distanceMatrix)

        # 5.3 | check if newSol is better than currentSol (note that we are working with a minimization problem)
        deltaObjective = newObjective - currentObjective
        if deltaObjective < 0:
            currentSol = np.array(newSol)
            currentObjective = np.array(newObjective)
            if currentObjective < bestObjective:
                bestObjective = np.array(currentObjective)
                bestSol = np.array(currentSol)
            nonImprovementCounter = int(0)
        else:
            randomNum = np.random.random()
            if randomNum < math.exp(-deltaObjective/temperatureNow):
                currentObjective = np.array(newObjective)
                currentSol = np.array(newSol)
            nonImprovementCounter += 1

        # 5.4 | storing the current solution
        solutionRepo.append(bestObjective)

        # 5.5 | checking the termination conditions and update the iterationCount    
        if nonImprovementCounter >= maxNonImprove:
            break
        else:
            iterationCount = iterationCount+1
            if iterationCount == maxIteration:
                temperatureNow = temperatureNow*coolingRate
                iterationCount = int(1)
    runtimeCount = toc()
    
    return bestSol, bestObjective, solutionRepo, runtimeCount

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple TSP case with 1 depot and 10 customer nodes

# symmetric distance matrix [11 x 11]
distanceMatrix = np.array([
[0.000, 2.768, 7.525, 9.689, 28.045, 36.075, 25.754, 3.713, 2.701, 8.286, 7.944],
[2.768, 0.000, 9.164, 11.482, 27.779, 37.431, 25.488, 7.305, 1.928, 7.911, 10.758],
[7.525, 9.164, 0.000, 16.297, 33.406, 42.246, 31.115, 4.065, 8.062, 13.970, 1.594],
[9.689, 11.482, 16.297, 0.000, 19.053, 27.198, 16.762, 12.484, 8.810, 17.182, 20.464],
[28.045, 27.779, 33.406, 19.053, 0.000, 10.138, 5.596, 31.349, 26.286, 36.994, 38.820],
[36.075, 37.431, 42.246, 27.198, 10.138, 0.000, 10.634, 38.434, 34.759, 43.132, 46.850],
[25.754, 25.488, 31.115, 16.762, 5.596, 10.634, 0.000, 29.058, 23.995, 32.498, 36.529],
[3.713, 7.305, 4.065, 12.484, 31.349, 38.434, 29.058, 0.000, 6.005, 8.615, 5.651],
[2.701, 1.928, 8.062, 8.810, 26.286, 34.759, 23.995, 6.005, 0.000, 9.862, 9.656],
[8.286, 7.911, 13.970, 17.182, 36.994, 43.132, 32.498, 8.615, 9.862, 0.000, 15.313],
[7.944, 10.758, 1.594, 20.464, 38.820, 46.850, 36.529, 5.651, 9.656, 15.313, 0.000]])

# number of nodes on the graph, can be calculated as the horizontal/vertical length of the distance matrix
numNodes = len(distanceMatrix[0:])

#%% define parameters for the simulated annealing
# note that all of these numbers are selected arbitrarily
temperatureMax = int(150) # starting temperature
temperatureMin = float(0.01) # lower bound temperature
maxNonImprove = int(10000) # maximum number of iterations with non-improvement on the best solution found, to terminate
maxIteration = int(1000) # maximum iteration per temperature
coolingRate = int(0.99) # cooling rate of temperature

#%% implement the simulated annealing to solve TSP
# bestSol --> the TSP solution with best objective value
# bestObjective --> best objective value found
# solutionRepo --> a repository of best solutions found on each iteration (to see the convergence of the algorithm)
# runtimeCount --> return the runtime of the algorithm in seconds
bestSol, bestObjective, solutionRepo, runtimeCount = simulatedAnnealing(distanceMatrix, numNodes, temperatureMax, temperatureMin, maxNonImprove, maxIteration, coolingRate)