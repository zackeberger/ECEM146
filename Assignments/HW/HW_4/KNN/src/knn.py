#!/usr/bin/env python
"""Analysis of titanic dataset using K-NN algorithm"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Define class that implements K-NN algorithm
class Knear:
    def __init__(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        
        # If len(trainingData) != len(trainingLabels), throw error
        self.datamt = len(trainingData)

    # Return an array of predicted labels for testing_data
    def test(self, testing_data, k_value, y_tie):
        
        labels = []
        for i in range(0, len(testing_data)):
            val = self.testPoint(testing_data[i], k_value, y_tie)
            labels.append(val)
        
        return labels

    # Test an individual point and return its label
    def testPoint(self, point, k_value, y_tie):
        
        # Obtain array of Euclidian distances from point to training set
        distances = np.array([])
        for i in range(0, self.datamt):
            dist = euclid_six_distance(point, self.trainingData[i])
            distances = np.append(distances, dist)

        # Obtain the k lowest indices, which are the k nearest neighbors
        # Use mergesort, which is a stable sorting algorithm
        nearest_neighbors = distances.argsort(kind='stable')[:k_value]
        
        # Count the 1s and 0s in the k nearest neighbors
        ones = 0
        zeros = 0

        for i in range(0, len(nearest_neighbors)):
            if self.trainingLabels[nearest_neighbors[i]] == 1:
                ones += 1
            else:
                zeros += 1

        # Determine the label of the test point
        if ones > zeros:
            return 1
        elif zeros > ones:
            return 0
        else:
            return y_tie


def main():
 
    # Import all training and testing data
    training_data = np.loadtxt("../data/dataTraining_X.csv", delimiter=',')
    actual_training_labels = np.loadtxt("../data/dataTraining_Y.csv", delimiter=',')
    testing_data = np.loadtxt("../data/dataTesting_X.csv", delimiter=',')
    actual_testing_labels = np.loadtxt("../data/dataTesting_Y.csv", delimiter=',')
   
    # Instantiate K-NN framework with training data
    knn = Knear(training_data, actual_training_labels)
    

    # With tiebreaker set to 1, find testing error for k = 1, ..., 15
    tiebreaker = 1
   
    test1 = knn.test(testing_data, 1, tiebreaker)
    err1 = error(test1, actual_testing_labels)
   
    test2 = knn.test(testing_data, 2, tiebreaker)
    err2 = error(test2, actual_testing_labels)
    
    test3 = knn.test(testing_data, 3, tiebreaker) 
    err3 = error(test3, actual_testing_labels)
    
    test4 = knn.test(testing_data, 4, tiebreaker)
    err4 = error(test4, actual_testing_labels)
    
    test5 = knn.test(testing_data, 5, tiebreaker)
    err5 = error(test5, actual_testing_labels)
    
    test6 = knn.test(testing_data, 6, tiebreaker) 
    err6 = error(test6, actual_testing_labels)

    test7 = knn.test(testing_data, 7, tiebreaker)
    err7 = error(test7, actual_testing_labels)
   
    test8 = knn.test(testing_data, 8, tiebreaker)
    err8 = error(test8, actual_testing_labels)
    
    test9 = knn.test(testing_data, 9, tiebreaker) 
    err9 = error(test9, actual_testing_labels)
    
    test10 = knn.test(testing_data, 10, tiebreaker)
    err10 = error(test10, actual_testing_labels)
    
    test11 = knn.test(testing_data, 11, tiebreaker)
    err11 = error(test11, actual_testing_labels)
    
    test12 = knn.test(testing_data, 12, tiebreaker) 
    err12 = error(test12, actual_testing_labels)

    test13 = knn.test(testing_data, 13, tiebreaker)
    err13 = error(test13, actual_testing_labels)
   
    test14 = knn.test(testing_data, 14, tiebreaker)
    err14 = error(test14, actual_testing_labels)
    
    test15 = knn.test(testing_data, 15, tiebreaker) 
    err15 = error(test15, actual_testing_labels)
    

    errors = [err1, err2, err3, err4, err5, err6, err7, err8, err9, err10, err11, err12, err13, err14, err15]
    print('Testing Error:')
    print(errors)

    # With tiebreaker set to 1, find training error for k = 1, ..., 15
    
    train1 = knn.test(training_data, 1, tiebreaker)
    errb1 = error(train1, actual_training_labels)

    train2 = knn.test(training_data, 2, tiebreaker)
    errb2 = error(train2, actual_training_labels)

    train3 = knn.test(training_data, 3, tiebreaker)
    errb3 = error(train3, actual_training_labels)

    train4 = knn.test(training_data, 4, tiebreaker)
    errb4 = error(train4, actual_training_labels)

    train5 = knn.test(training_data, 5, tiebreaker)
    errb5 = error(train5, actual_training_labels)

    train6 = knn.test(training_data, 6, tiebreaker)
    errb6 = error(train6, actual_training_labels)

    train7 = knn.test(training_data, 7, tiebreaker)
    errb7 = error(train7, actual_training_labels)

    train8 = knn.test(training_data, 8, tiebreaker)
    errb8 = error(train8, actual_training_labels)

    train9 = knn.test(training_data, 9, tiebreaker)
    errb9 = error(train9, actual_training_labels)

    train10 = knn.test(training_data, 10, tiebreaker)
    errb10 = error(train10, actual_training_labels)

    train11 = knn.test(training_data, 11, tiebreaker)
    errb11 = error(train11, actual_training_labels)

    train12 = knn.test(training_data, 12, tiebreaker)
    errb12 = error(train12, actual_training_labels)

    train13 = knn.test(training_data, 13, tiebreaker)
    errb13 = error(train13, actual_training_labels)

    train14 = knn.test(training_data, 14, tiebreaker)
    errb14 = error(train14, actual_training_labels)

    train15 = knn.test(training_data, 15, tiebreaker)
    errb15 = error(train15, actual_training_labels)

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    errorbs = [errb1, errb2, errb3, errb4, errb5, errb6, errb7, errb8, errb9, errb10, errb11, errb12, errb13, errb14, errb15]
    print('Training Error:')
    print(errorbs)

    plt.scatter(x, errors, color='r', label='Testing Error')
    plt.scatter(x, errorbs, color='b', label='Training Error')
    plt.title('Training & Testing Error with Y_tie = 1')
    plt.xlabel('K Value')
    plt.ylabel('Error')
    plt.legend(loc="upper right")
    plt.show() 

    
    """
    # With tiebreaker set to 0, find testing error for k = 1, ..., 15
    tiebreaker = 0

    test1 = knn.test(testing_data, 1, tiebreaker)
    err1 = error(test1, actual_testing_labels)
   
    test2 = knn.test(testing_data, 2, tiebreaker)
    err2 = error(test2, actual_testing_labels)
    
    test3 = knn.test(testing_data, 3, tiebreaker) 
    err3 = error(test3, actual_testing_labels)
    
    test4 = knn.test(testing_data, 4, tiebreaker)
    err4 = error(test4, actual_testing_labels)
    
    test5 = knn.test(testing_data, 5, tiebreaker)
    err5 = error(test5, actual_testing_labels)
    
    test6 = knn.test(testing_data, 6, tiebreaker) 
    err6 = error(test6, actual_testing_labels)

    test7 = knn.test(testing_data, 7, tiebreaker)
    err7 = error(test7, actual_testing_labels)
   
    test8 = knn.test(testing_data, 8, tiebreaker)
    err8 = error(test8, actual_testing_labels)
    
    test9 = knn.test(testing_data, 9, tiebreaker) 
    err9 = error(test9, actual_testing_labels)
    
    test10 = knn.test(testing_data, 10, tiebreaker)
    err10 = error(test10, actual_testing_labels)
    
    test11 = knn.test(testing_data, 11, tiebreaker)
    err11 = error(test11, actual_testing_labels)
    
    test12 = knn.test(testing_data, 12, tiebreaker) 
    err12 = error(test12, actual_testing_labels)

    test13 = knn.test(testing_data, 13, tiebreaker)
    err13 = error(test13, actual_testing_labels)
   
    test14 = knn.test(testing_data, 14, tiebreaker)
    err14 = error(test14, actual_testing_labels)
    
    test15 = knn.test(testing_data, 15, tiebreaker) 
    err15 = error(test15, actual_testing_labels)
    

    errors = [err1, err2, err3, err4, err5, err6, err7, err8, err9, err10, err11, err12, err13, err14, err15]
    print('Testing Error:')
    print(errors)

    
    # With tiebreaker set to 0, find training error for k = 1, ..., 15
    train1 = knn.test(training_data, 1, tiebreaker)
    errb1 = error(train1, actual_training_labels)

    train2 = knn.test(training_data, 2, tiebreaker)
    errb2 = error(train2, actual_training_labels)

    train3 = knn.test(training_data, 3, tiebreaker)
    errb3 = error(train3, actual_training_labels)

    train4 = knn.test(training_data, 4, tiebreaker)
    errb4 = error(train4, actual_training_labels)

    train5 = knn.test(training_data, 5, tiebreaker)
    errb5 = error(train5, actual_training_labels)

    train6 = knn.test(training_data, 6, tiebreaker)
    errb6 = error(train6, actual_training_labels)

    train7 = knn.test(training_data, 7, tiebreaker)
    errb7 = error(train7, actual_training_labels)

    train8 = knn.test(training_data, 8, tiebreaker)
    errb8 = error(train8, actual_training_labels)

    train9 = knn.test(training_data, 9, tiebreaker)
    errb9 = error(train9, actual_training_labels)

    train10 = knn.test(training_data, 10, tiebreaker)
    errb10 = error(train10, actual_training_labels)

    train11 = knn.test(training_data, 11, tiebreaker)
    errb11 = error(train11, actual_training_labels)

    train12 = knn.test(training_data, 12, tiebreaker)
    errb12 = error(train12, actual_training_labels)

    train13 = knn.test(training_data, 13, tiebreaker)
    errb13 = error(train13, actual_training_labels)

    train14 = knn.test(training_data, 14, tiebreaker)
    errb14 = error(train14, actual_training_labels)

    train15 = knn.test(training_data, 15, tiebreaker)
    errb15 = error(train15, actual_training_labels)

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    errorbs = [errb1, errb2, errb3, errb4, errb5, errb6, errb7, errb8, errb9, errb10, errb11, errb12, errb13, errb14, errb15]
    print('Training Error:')
    print(errorbs)
    
    plt.scatter(x, errors, color='r', label='Testing Error')
    plt.scatter(x, errorbs, color='b', label='Training Error')
    plt.title('Training & Testing Error with Y_tie = 0')
    plt.xlabel('K Value')
    plt.ylabel('Error')
    plt.legend(loc="lower right")
    plt.show()
    """


# Compute percentage of mismatches in two equivalently sized arrays
def error(arr1, arr2):
    matches = 0

    for i in range(0, len(arr1)):
        if arr1[i] == arr2[i]:
            matches += 1

    return 1 - ( float(matches) / len(arr1) )


# Find the Euclidian distance between two points with six attributes
def euclid_six_distance(arr1, arr2):

    return ( (arr1[0] - arr2[0])**2 + (arr1[1] - arr2[1])**2 + (arr1[2] - arr2[2])**2 + (arr1[3] - arr2[3])**2 + (arr1[4] - arr2[4])**2 + (arr1[5] - arr2[5])**2 )


# Name guard
if __name__ == "__main__":
    main()
