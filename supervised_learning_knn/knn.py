'''
This script aims to implement k-nearest neighbors algorithm to merged_accident.csv
in order to predict the severity of accidents based on a given condition (road surface/ atmospheric condition)

The script includes the following steps:
1. Separate the data into training and testing sets
2. Separate the training data into training and validation sets
3. Train KNN model with different values of k
4. Run each model on the validation set
5. Determine the accuracy, precision, recall, and F1 score of each model
6. Plot the results (accuracy, precision, recall, and F1 score) for each k value
7. Determine the best k value
8. Train the KNN model with the best k value on entire training set
9. Run the model on test set
10. Determine the accuracy, precision, recall, and F1 score of the model on test set
11. Plot the confusion matrix
12. Save the model and all graphs
13. Conclude the results
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# main function
def main():
    # load dataset
    data = pd.read_csv('merged_accident.csv')

    # only keep columns of interest
    given_cond = input("Please enter the condition you want to use (surface/ atmosphere): ")
    if given_cond == 'surface':
        data = data[['SURFACE_INDEX', 'SEVERITY']]
    elif given_cond == 'atmosphere':
        data = data[['ATMOSPH_INDEX', 'SEVERITY']]
    else:
        print("Invalid condition. Please enter either 'surface' or 'atmosphere'.")
        return
    
    # separate data into training, validation, and test sets







