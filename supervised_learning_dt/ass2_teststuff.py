# Supervised Machine Learning 1: Decision Tree Classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_score

# Remap severity to flip order
# read merged_accident.csv

merged_df = pd.read_csv('merged_accident.csv')

merged_df['SEVERITY'] = merged_df['SEVERITY'].apply(lambda x: 5 - x)

""" # temporarily removed bc imma try stratified sampling
# Split data into 80/20 train/test
train_size = int(0.8 * len(merged_df))
train = merged_df[:train_size]
test = merged_df[train_size:]

train_subset_size = int(0.8*len(train))
train_subset = train[:train_subset_size]
validation = train[train_subset_size:]
"""

# Split using stratified sampling
train, test = train_test_split(merged_df, test_size =0.2, stratify = merged_df['SEVERITY'], random_state=42)    

# Further split train into train_subset and validation
train_subset, validation = train_test_split(train, test_size =0.2, stratify = train['SEVERITY'], random_state=42)

# Export csvs
#train.to_csv('train.csv', index=False)
#test.to_csv('test.csv', index=False)
#train_subset.to_csv('train_subset.csv', index=False)
#validation.to_csv('validation.csv', index=False)

""" # Temporarily removed bc i think i might not need ordinal encoding
# Perform ordinal encoding
X_train_subset = OrdinalEncoder().fit_transform(train_subset[['ATMOSPH_INDEX', 'SURFACE_INDEX']])
y_train_subset = OrdinalEncoder().fit_transform(train_subset[['SEVERITY']])[:, 0] 

X_validation = OrdinalEncoder().fit_transform(validation[['ATMOSPH_INDEX', 'SURFACE_INDEX']])
y_validation = OrdinalEncoder().fit_transform(validation[['SEVERITY']])[:, 0] 
"""

X_train_subset = train_subset[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
y_train_subset = train_subset['SEVERITY']

X_validation = validation[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
y_validation = validation['SEVERITY']


# Want to know if average of macro is better, use .value_counts() on train_subset to see how one sided data is
# Count the number of occurrences in 'ATMOSPH_INDEX' and 'SURFACE_INDEX'
atm_index_counts = train_subset['ATMOSPH_INDEX'].value_counts()
surface_index_counts = train_subset['SURFACE_INDEX'].value_counts()
# Ok theres way more 1.0 and 3.0 so macro is better (I DIDN'T LOOK AT TRAIN DATA)

#severity_vals = train_subset['SEVERITY'].value_counts()
#print(severity_vals)

# Print the counts
#print("Count of instances for ATMOSPH_INDEX:")
#print(atm_index_counts)
#print("\nCount of instances for SURFACE_INDEX:")
#print(surface_index_counts)

# Hyperparameter selection
depths = list(range(1,10))
f1_list = []
accuracy_list = []
precision_list = []

for depth in depths:
    d_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, class_weight= 'balanced')
    d_tree.fit(X_train_subset, y_train_subset)
    y_pred = d_tree.predict(X_validation)
    f1 = f1_score(y_validation, y_pred, average='weighted')
    f1_list.append(f1)
    accuracy = accuracy_score(y_validation, y_pred)
    accuracy_list.append(accuracy)
    precision = precision_score(y_validation, y_pred, average='weighted', zero_division=0)
    precision_list.append(precision)

# Determine optimal depth
max_f1 = max(f1_list)
optimal_depth = depths[f1_list.index(max_f1)]
print(f"\nMax f1 score: {max(f1_list)}, at depth: {depths[f1_list.index(max(f1_list))]}\n")

# Determine max accuracy
max_accuracy = max(accuracy_list)
optimal_accuracy_depth = depths[accuracy_list.index(max_accuracy)]
print(f"\nMax accuracy: {max(accuracy_list)}, at depth: {depths[accuracy_list.index(max_accuracy)]}\n")

# Determine max precision
max_precision = max(precision_list)
optimal_precision_depth = depths[precision_list.index(max_precision)]
print(f"\nMax precision: {max(precision_list)}, at depth: {depths[precision_list.index(max_precision)]}\n")

# # # Plot f1 scores for depth (x) and f1 score (y)
# plt.plot(depths, f1_list)
# plt.xlabel('Depth')
# plt.ylabel('F1 Score')
# plt.title('F1 Score vs Depth')
# plt.xticks(depths)
# plt.grid(True)
# plt.show()

# # # Plot accuracy for depth (x) and accuracy (y)
# plt.plot(depths, accuracy_list)
# plt.xlabel('Depth')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Depth')
# plt.xticks(depths)
# plt.grid(True)
# plt.show()

# # Retrain with full training set and optimal depth
# X_train = train[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
# y_train = train['SEVERITY']

# final_d_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=optimal_depth, class_weight= 'balanced')
# final_d_tree.fit(X_train, y_train)

# # Plot tree
# plt.figure(figsize=(20, 10))
# plot_tree(final_d_tree, filled=True, feature_names=['ATMOSPH_INDEX', 'SURFACE_INDEX'], class_names=['Non-injury', 'Minor', 'Serious', 'Fatal'], fontsize= 7)
# plt.title("Decision Tree Classifier")
# plt.show()

# train_pred = final_d_tree.predict(X_train)
# train_f1 = f1_score(y_train, train_pred, average = 'weighted')

# accuracy = accuracy_score(y_train, train_pred)
# print(f"Accuracy on train set: {accuracy:.4f}\n")

# print(classification_report(y_train, train_pred, target_names=['Non-injury', 'Minor', 'Serious', 'Fatal']))

# # Test on test set
# X_test = test[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
# y_test = test['SEVERITY']

# test_pred = final_d_tree.predict(X_test)
# test_f1 = f1_score(y_test, test_pred, average = 'weighted')

# # Plot confusion matrix
# cm = confusion_matrix(y_test, test_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-injury', 'Minor', 'Serious', 'Fatal'])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()

# accuracy = accuracy_score(y_test, test_pred)
# print(f"Accuracy on test set: {accuracy:.4f}\n")

# print(f"\nF1 score on test set: {test_f1}, F1 score on train set: {train_f1}\n")
# print(classification_report(y_test, test_pred, target_names=['Non-injury', 'Minor', 'Serious', 'Fatal']))

print("hello")
print("world")

