#!/usr/bin/env python
"""Analysis of the titanic dataset using a decision tree classifier"""

import numpy as np
from sklearn import tree

def main():

    # Import all training and testing data
    training_dataX = np.loadtxt("../data/dataTraining_X.csv", delimiter=',') 
    training_dataY = np.loadtxt("../data/dataTraining_Y.csv", delimiter=',')
    testing_dataX = np.loadtxt("../data/dataTesting_X.csv", delimiter=',')
    testing_dataY = np.loadtxt("../data/dataTesting_Y.csv", delimiter=',')
    
    # Basline classification
    baseline_label = baseline_classification(training_dataY)
    
    correct_training_predictions = 0
    for i in range (0, len(training_dataY)):
        if training_dataY[i] == baseline_label:
            correct_training_predictions += 1

    correct_testing_predictions = 0
    for i in range (0, len(testing_dataY)):
        if testing_dataY[i] == baseline_label:
            correct_testing_predictions += 1

    # Compute training and testing accuracy
    print("baseline label: " + str(baseline_label) )
    print("training set size: " + str(len(training_dataY)) + ". correct training predictions: " + str(correct_training_predictions) )
    print("testing set size: " + str(len(testing_dataY)) + ". correct testing predictions: " + str(correct_testing_predictions) )

    accuracy_training = float(correct_training_predictions) / len(training_dataY)
    accuracy_testing = float(correct_testing_predictions) / len(testing_dataY)

    print("Accuracy of baseline classifier on training set: " + str(accuracy_training))
    print("Accuracy of baseline classifier on testing set: " + str(accuracy_testing) + "\n")
    
    # Train a scikit decision tree
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(training_dataX, training_dataY)

    # Test the decision tree
    train_predictions = clf.predict(training_dataX)
    correct_training_predictions = count_matches(train_predictions, training_dataY)
    accuracy_training = float(correct_training_predictions) / len(train_predictions)

    test_predictions = clf.predict(testing_dataX)
    correct_testing_predictions = count_matches(test_predictions, testing_dataY)
    accuracy_testing = float(correct_testing_predictions) / len(test_predictions)

    print("Accuracy of trained decision tree on training set: " + str(accuracy_training))
    print("Accuracy of trained decision tree on testing set: " + str(accuracy_testing))



# Count the number of matches between two arrays of equal length
def count_matches(arr1, arr2):
    
    matches = 0
    for i in range (0, len(arr1)):
        if arr1[i] == arr2[i]:
            matches += 1
    
    return matches



# Obtain baseline classification result by majority count
# Return the value (1 or 0) that appears the most in the array
def baseline_classification(arr):
    
    one_count = 0
    zero_count = 0

    # Obtain how many zeros vs. ones there are in the set
    for i in range (0,len(arr)):
        if arr[i] == 0:
            zero_count += 1
        else:
            one_count += 1
    
    # Return the majority
    if one_count > zero_count:
        majority = 1
    else:
        majority = 0

    return majority



# Name guard
if __name__ == "__main__":
    main()
