import numpy as np
import random
import matplotlib.pyplot as plt

# Functions
def awjm(matrix,candidates,variables):
    
    fx = np.empty(candidates, dtype=float)
    for i in range(candidates):
        y1 = -1.15146 + 0.70118 * matrix[i][0] + 2.72749 * matrix[i][1] + 0.00689 * matrix[i][2] - 0.00025 * matrix[i][3] + 0.00386 * matrix[i][1] * matrix[i][2]
        y2 = -0.93947 * matrix[i][1] * matrix[i][1] - 0.25711 * matrix[i][0] * matrix[i][1] - 0.00314 * matrix[i][0] * matrix[i][2] - 0.00249 * matrix[i][0] * matrix[i][3] + 0.00196 * matrix[i][1] * matrix[i][3] - 0.00002 * matrix[i][2] * matrix[i][3]
        fx[i] = y1+y2
    
    print("function Values : ", fx)
    return fx
    
def probability(fx):
    
    probabilityValues = np.empty(candidates)
    for i in range(candidates):
        probabilityValues[i] = (1/fx[i]) / (sum(np.reciprocal(fx)))
    # print("Probability Values : ", probabilityValues)
    return probabilityValues 

def candidatefollowMatrix(probabilityValues,candidates,matrix):
    
    newMatrix = np.empty([candidates, variables])
    probabilityLine = [sum(probabilityValues[0:x:1]) for x in range(candidates + 1)]
    randomProbabilityValues = np.empty(candidates)
    followMatrix = []

    # Generating random probabilities
    for _ in range(candidates):
        randomProbabilityValues[_] = random.uniform(0, 1)
        
    # Generating follow matrix
    for i in randomProbabilityValues:
        for j in range(len(probabilityLine)-1):
            if probabilityLine[j] < i <= probabilityLine[j+1]:
                followMatrix.append(j) 

    # leader = np.argmax(randomProbabilityValues)
    # followMatrix[0] = leader
    #Generating new matrix
    for i in range(candidates):
        newMatrix[i] = matrix[followMatrix[i]]

    return newMatrix

def updateSamplingIntervals(matrix, newMatrix, upperbounds, lowerbounds, originalUpperbounds, originalLowerbounds, r):

    for i in range(variables):
        newRange = upperbounds[i] - lowerbounds[i] 
        newRange = newRange * r
        upperbounds[i] = newRange / 2
        lowerbounds[i] = -newRange / 2

    for x in range(candidates):
        for y in range(variables):
            upper = newMatrix[x][y] + upperbounds[y]
            lower = newMatrix[x][y] + lowerbounds[y]
            if upper > originalUpperbounds[y]:
                upper = originalUpperbounds[y]
            if lower < originalLowerbounds[y]:
                lower = originalLowerbounds[y]    
            matrix[x][y] = random.uniform(upper,lower)       
    return matrix

if __name__ == "__main__":

    # Initializations
    candidates = int(input("Number of candidates : "))
    variables =  int(input("Number of variables : "))
    r = 0.9
    matrix = np.empty([candidates, variables])
    upperbounds = [1.25, 1.5, 96, 600]     
    lowerbounds = [0.9, 0.95, 20, 200]
    originalUpperbounds = [1.25, 1.5, 96, 600]     
    originalLowerbounds = [0.9, 0.95, 20, 200]

    # Generating initial matrix
    for k in range(candidates):
        for h in range(variables):
            matrix[k][h] = random.uniform(upperbounds[h], lowerbounds[h])
            
    x_graph = []
    y_graph = []

    for i in range(500):
        fx = awjm(matrix, candidates, variables)
        probabilityValues = probability(fx)
        newMatrix = candidatefollowMatrix(probabilityValues,candidates,matrix)
        updatedIntervals = updateSamplingIntervals(matrix, newMatrix, upperbounds, lowerbounds, originalUpperbounds, originalLowerbounds, r)
        x_graph.append(i)
        y_graph.append(fx)
    plt.plot(x_graph, y_graph)
    plt.xlabel("Iterations")
    plt.ylabel("Function Values")
    plt.show()
