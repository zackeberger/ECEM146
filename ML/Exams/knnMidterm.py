#!/usr/bin/env python
"""Analysis of midterm dataset using K-NN algorithm"""

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
        
        # Obtain array of l one distances from point to training set
        distances = np.array([])
        for i in range(0, self.datamt):
            dist = l_one_distance(point, self.trainingData[i])
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
 
    # Import and parse all training and testing data
    # Because alpha = 2, want 11th to 20th rows as testing, rest as training
    data = np.loadtxt("/Users/zackberger/Desktop/Ml/Exams/Q2data.csv", delimiter=',')
  
    training_info = np.concatenate([ data[0:10], data[20:] ])
    testing_info = data[10:20]
    
    training_data = training_info[:,:2]
    training_labels = training_info[:,2]
    
    testing_data = testing_info[:,:2]
    testing_labels = testing_info[:,2]

    # Plot data
    for i in range(0, len(training_data)):
        if training_labels[i] == 1:
            plt.scatter(training_data[i,0], training_data[i,1], color='r')
        else:
            plt.scatter(training_data[i,0], training_data[i,1], color='b')

    plt.scatter(testing_data[:,0], testing_data[:,1], color='cyan')
    plt.title('Plot of Training and Testing Data')
    plt.xlabel("x1 Value")
    plt.ylabel("x2 Value")
    plt.legend(loc="upper right")
    plt.show()

    # Instantiate K-NN framework with training data
    knn = Knear(training_data, training_labels)
    
    # For even k, if number of points from class 0 is same as number of points from class 1, classify
    # this data point as class 0 deterministically by setting the parameter tiebreaker = 0
    tiebreaker = 0
   
    test1 = knn.test(testing_data, 1, tiebreaker)
    acc1 = testing_accuracy(test1, testing_labels)
   
    test2 = knn.test(testing_data, 2, tiebreaker)
    acc2 = testing_accuracy(test2, testing_labels)
    
    test3 = knn.test(testing_data, 3, tiebreaker) 
    acc3 = testing_accuracy(test3, testing_labels)
    
    test4 = knn.test(testing_data, 4, tiebreaker)
    acc4 = testing_accuracy(test4, testing_labels)
    
    test5 = knn.test(testing_data, 5, tiebreaker)
    acc5 = testing_accuracy(test5, testing_labels)
    
    test6 = knn.test(testing_data, 6, tiebreaker) 
    acc6 = testing_accuracy(test6, testing_labels)

    test7 = knn.test(testing_data, 7, tiebreaker)
    acc7 = testing_accuracy(test7, testing_labels)
   
    test8 = knn.test(testing_data, 8, tiebreaker)
    acc8 = testing_accuracy(test8, testing_labels)
    
    test9 = knn.test(testing_data, 9, tiebreaker) 
    acc9 = testing_accuracy(test9, testing_labels)
 
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    accuracies = [acc1, acc2, acc3, acc4, acc5,  acc6, acc7, acc8, acc9] 
    print('Testing Accuracies:')
    print(accuracies)

    plt.scatter(x, accuracies, color='g', label='Testing Accuracy')
    plt.title('Testing Accuracy for k = 1, ..., 9')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper right")
    plt.show() 

    

# Compute percentage of matches in two equivalently sized arrays
def testing_accuracy(arr1, arr2):
    matches = 0

    for i in range(0, len(arr1)):
        if arr1[i] == arr2[i]:
            matches += 1

    return ( float(matches) / len(arr1) )


# Find the l_one distance between two pointss with two attributes
def l_one_distance(arr1, arr2):

    return ( abs(arr1[0] - arr2[0]) + abs(arr1[1] - arr2[1]) )


# Name guard
if __name__ == "__main__":
    main()
