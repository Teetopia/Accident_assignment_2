# Supervised Machine Learning 1: Decision Tree Classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score

# Remap severity to flip order
# read merged_accident.csv

merged_df = pd.read_csv('merged_accident.csv')

merged_df['SEVERITY'] = merged_df['SEVERITY'].apply(lambda x: 5 - x)
    
# Split data into 80/20 train/test
train_size = int(0.8 * len(merged_df))
train = merged_df[:train_size]
test = merged_df[train_size:]

train_subset_size = int(0.8*len(train))
train_subset = train[:train_subset_size]
validation = train[train_subset_size:]

# Export csvs
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
train_subset.to_csv('train_subset.csv', index=False)
validation.to_csv('validation.csv', index=False)

# Perform ordinal encoding
X_train_subset = OrdinalEncoder().fit_transform(train_subset[['ATMOSPH_INDEX', 'SURFACE_INDEX']])
y_train_subset = OrdinalEncoder().fit_transform(train_subset[['SEVERITY']])[:, 0] 

X_validation = OrdinalEncoder().fit_transform(validation[['ATMOSPH_INDEX', 'SURFACE_INDEX']])
y_validation = OrdinalEncoder().fit_transform(validation[['SEVERITY']])[:, 0] 

# Want to know if average or macro is better, use .value_counts() on train_subset to see how one sided data is
# Count the number of occurrences in 'ATMOSPH_INDEX' and 'SURFACE_INDEX'
atm_index_counts = train_subset['ATMOSPH_INDEX'].value_counts()
surface_index_counts = train_subset['SURFACE_INDEX'].value_counts()
# Ok theres way more 1.0 and 3.0 so macro is better (I DIDN'T LOOK AT TRAIN DATA)

severity_vals = train_subset['SEVERITY'].value_counts()
print(severity_vals)

# Print the counts
#print("Count of instances for ATMOSPH_INDEX:")
#print(atm_index_counts)
#print("\nCount of instances for SURFACE_INDEX:")
#print(surface_index_counts)

# Hyperparameter selection
depths = list(range(1, 5))
f1_list = []

for depth in depths:
    test_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, class_weight='balanced')
    test_tree.fit(X_train_subset, y_train_subset)
    y_pred = test_tree.predict(X_validation)
    f1 = f1_score(y_validation, y_pred, average='macro')
    f1_list.append(f1)

# Determine optimal depth
max_f1 = max(f1_list)
optimal_depth = depths[f1_list.index(max_f1)]
# print(f"\nMax f1 score: {max(f1_list)}, at depth: {depths[f1_list.index(max(f1_list))]}")

# Retrain with full training set and optimal depth
X_train = OrdinalEncoder().fit_transform(train[['ATMOSPH_INDEX', 'SURFACE_INDEX']])
y_train = OrdinalEncoder().fit_transform(train[['SEVERITY']])[:, 0]

test_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=optimal_depth, class_weight='balanced')
test_tree.fit(X_train, y_train)

# Plot tree
plt.figure(figsize=(20, 10))
plot_tree(test_tree, filled=True, feature_names=['ATMOSPH_INDEX', 'SURFACE_INDEX'], class_names=['Non-injury', 'Minor', 'Serious', 'Fatal'], fontsize= 4)
plt.title("Decision Tree Classifier")
plt.show()


