'''
This script aims to implement k-nearest neighbors algorithm to merged_accident.csv
in order to predict the severity of accidents based on a given condition (road surface/ atmospheric condition)

The script includes the following steps:
1. Separate the data into training and testing sets
2a. Separate the training data into training and validation sets
2b. Perform stratified sampling to train sets with similar number of each class
3. Train KNN model with different values of k
4. Run each model on the validation set
5. Determine the accuracy, precision, recall, and F1 score of each model
6. Plot the results (accuracy and F1 score) for each k value
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
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report



def separate_data(X, y):
    '''
    takes in pandas series of feature X and label y
    returns the training and testing sets in form of nparrays
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

    X_train = np.asarray(X_train).reshape(-1, 1)
    X_test = np.asarray(X_test).reshape(-1, 1)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, X_test, y_train, y_test

def try_ks(X_train, y_train, X_val, y_val):

    k_vals = range(1, 10)
    accuracies = []
    f1_scores = []

    for k in k_vals:
        knn = KNN(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

        print(f'k={k}\n\taccuracy={accuracies[-1]:.4f}\n\tf1_score={f1_scores[-1]:.4f}')
        print(classification_report(y_val, y_pred, zero_division=0))

    # find best f1 score and its corresponding k value and accuracy
    best_f1 = max(f1_scores)
    try_ks = k_vals[f1_scores.index(best_f1)]
    best_accuracy = accuracies[f1_scores.index(best_f1)]
    print(f'Best k value: {try_ks}, F1 Score: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}')

def best_k(X_train, y_train):
    # Method 1: oversample the minority class
    # oversample the minority class
    ros = RandomOverSampler(random_state=17)
    X_train_os, y_train_os = ros.fit_resample(np.asarray(X_train).reshape(-1, 1), np.asarray(y_train))

    # separate the oversampled data into training and validation sets
    X_train_os, X_val_os, y_train_os, y_val_os = separate_data(pd.Series(X_train_os.flatten()), pd.Series(y_train_os.flatten()))

    # Method 2: keep test set as is
    # separate the training data into training and validation sets
    X_train, X_val, y_train, y_val = separate_data(X_train, y_train)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    scaler_os = StandardScaler().fit(X_train_os)
    X_train_os = scaler_os.transform(X_train_os)
    X_val_os = scaler_os.transform(X_val_os)

    # determine best k value for imbalanced data
    print('Original data(imbalanced):')
    try_ks(X_train, y_train, X_val, y_val)

    # determine best k value for oversampled data
    print('Balanced data(oversampled class 1 and 4):')
    try_ks(X_train_os, y_train_os, X_val_os, y_val_os)

    return

# main function
def main():
    # load dataset
    data = pd.read_csv('merged_accident.csv')

    # only keep columns of interest
    Xcol = input("Please enter the condition you want to use (s for surface/ a for atmosphere): ")
    if Xcol == 's':
        Xcol = 'SURFACE_INDEX'
        data = data[['SURFACE_INDEX', 'SEVERITY']]
    elif Xcol == 'a':
        Xcol = 'ATMOSPH_INDEX'
        data = data[['ATMOSPH_INDEX', 'SEVERITY']]
    else:
        print("Invalid condition. Please enter either 's' or 'a'.")

    X = data[Xcol]
    y = data['SEVERITY']

    # separate data into training and testing sets
    # do not touch the test set
    X_train, X_test, y_train, y_test = separate_data(X, y)

    # determine best k using the training set
    # best_k(X_train, y_train)
    # from the result, k=5 is chosen as it has the best f1 score given similar accuracy

    # train the KNN model with the best k value on the entire training set
    k = 5
    knn = KNN(n_neighbors=k)
    knn.fit(X_train, y_train)

    # run the model on the test set
    y_pred = knn.predict(X_test)

    # make clas



    
    return

main()







