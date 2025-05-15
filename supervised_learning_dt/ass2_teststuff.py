# Supervised Machine Learning 1: Decision Tree Classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

def get_acc_comparison(X_train, y_train, X_val, y_val):
    depths = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    acc_macro = []
    acc_weighted = []

    for depth in depths:
        # Case 1: No imbalance adjustment
        d_tree_macro = DecisionTreeClassifier(criterion='entropy', max_depth=depth, class_weight=None)
        d_tree_macro.fit(X_train, y_train)
        y_pred_macro = d_tree_macro.predict(X_val)
        acc_macro_score = accuracy_score(y_val, y_pred_macro)
        acc_macro.append(acc_macro_score)

        # Case 2: With imbalance adjustment
        d_tree_weighted = DecisionTreeClassifier(criterion='entropy', max_depth=depth, class_weight='balanced')
        d_tree_weighted.fit(X_train, y_train)
        y_pred_weighted = d_tree_weighted.predict(X_val)
        acc_weighted_score = accuracy_score(y_val, y_pred_weighted)
        acc_weighted.append(acc_weighted_score)

    # Print best scores
    best_macro = max(acc_macro)
    best_macro_depth = depths[acc_macro.index(best_macro)]
    best_weighted = max(acc_weighted)
    best_weighted_depth = depths[acc_weighted.index(best_weighted)]

    print(f"\nMax macro accuracy score: {best_macro:.4f} at depth {best_macro_depth}")
    print(f"Max weighted accuracy score: {best_weighted:.4f} at depth {best_weighted_depth}\n")

    # Plot both lines
    plt.plot(depths, acc_macro, label='Macro Avg (No Weight)', marker='o')
    plt.plot(depths, acc_weighted, label='Weighted Avg (Balanced)', marker='s')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score vs Tree Depth')
    plt.xticks(depths)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# Helper functions
def get_depth(X_train, y_train, X_val, y_val, imbalance_adjusted):
    depths = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    f1_list = []

    for depth in depths:
        d_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, class_weight= None if not imbalance_adjusted else 'balanced')
        d_tree.fit(X_train_subset, y_train_subset)
        y_pred = d_tree.predict(X_validation)
        f1 = f1_score(y_validation, y_pred, average= 'macro' if not imbalance_adjusted else 'weighted')
        f1_list.append(f1)
    
    # Determine optimal depth
    max_f1 = max(f1_list)
    optimal_depth = depths[f1_list.index(max_f1)]
    print(f"\nMax f1 score: {max(f1_list)}, at depth: {depths[f1_list.index(max(f1_list))]}\n")

    # Plot graph of depth (x) and f1 score (y)
    plt.plot(depths, f1_list)
    plt.xlabel('Depth')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Depth')
    plt.xticks(depths)
    plt.grid(True)
    plt.show()   
    
#     return optimal_depth

def train_and_test(X_train, y_train, X_test, y_test, depth, imbalance_adjusted):
    dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth, class_weight= None if not imbalance_adjusted else 'balanced')
    dt_model.fit(X_train, y_train)

    train_pred = dt_model.predict(X_train)
    train_f1 = f1_score(y_train, train_pred, average = 'macro' if not imbalance_adjusted else 'weighted')  

    # Plot tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, filled=True, feature_names=['ATMOSPH_INDEX', 'SURFACE_INDEX'], class_names=['Non-injury', 'Minor', 'Serious', 'Fatal'], fontsize= 7)
    plt.title("Decision Tree Classifier")
    plt.show()

    # Test on test set
    test_pred = dt_model.predict(X_test)
    test_f1 = f1_score(y_test, test_pred, average= 'macro' if not imbalance_adjusted else 'weighted')

    # Plot confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-injury', 'Minor', 'Serious', 'Fatal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report
    print(f"\nF1 score on test set: {test_f1}, F1 score on train set: {train_f1}\n")
    print(f"Accuracy on test set: {accuracy_score(y_test, test_pred)}, Accuracy on train set: {accuracy_score(y_train, train_pred)}\n")
    print("Classification report on train set:\n")
    print(classification_report(y_train, train_pred, target_names=['Non-injury', 'Minor', 'Serious', 'Fatal']))
    print("Classification report on test set:\n")
    print(classification_report(y_test, test_pred, target_names=['Non-injury', 'Minor', 'Serious', 'Fatal']))

# Remap severity to flip order
# read merged_accident.csv
merged_df = pd.read_csv('merged_accident.csv')

# Split using stratified sampling
train, test = train_test_split(merged_df, test_size =0.2, stratify = merged_df['SEVERITY'], random_state=42)    

# Further split train into train_subset and validation
train_subset, validation = train_test_split(train, test_size =0.2, stratify = train['SEVERITY'], random_state=42)

X_train_subset = train_subset[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
y_train_subset = train_subset['SEVERITY']

X_validation = validation[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
y_validation = validation['SEVERITY']

X_train = train[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
y_train = train['SEVERITY']
X_test = test[['ATMOSPH_INDEX', 'SURFACE_INDEX']]
y_test = test['SEVERITY']

# See what happens when imbalance is not accounted for at all
# Get optimal depth 
optimal_depth_imbal = get_depth(X_train_subset, y_train_subset, X_validation, y_validation, imbalance_adjusted = False)
# Train and test
train_and_test(X_train, y_train, X_test, y_test, optimal_depth_imbal, imbalance_adjusted = False)

# Now account for imbalance and run again
# Get optimal depth
optimal_depth_bal = get_depth(X_train_subset, y_train_subset, X_validation, y_validation, imbalance_adjusted = True)

# Train and test
train_and_test(X_train, y_train, X_test, y_test, optimal_depth_bal, imbalance_adjusted = True)

##################

# See what happens when there is only one independent variable tested at a time
atm_X_train_subset = train[['ATMOSPH_INDEX']]

atm_X_validation = validation[['ATMOSPH_INDEX']]

atm_X_train = train[['ATMOSPH_INDEX']]

atm_X_test = test[['ATMOSPH_INDEX']]


# See what happens when imbalance is not accounted for at all
# Get optimal depth 
atm_optimal_depth_imbal = get_depth(atm_X_train_subset, y_train_subset, atm_X_validation, y_validation, imbalance_adjusted = False)

# Train and test
train_and_test(atm_X_train, y_train, atm_X_test, y_test, atm_optimal_depth_imbal, imbalance_adjusted = False)

# Now account for imbalance and run again
# Get optimal depth
atm_optimal_depth_bal = get_depth(atm_X_train_subset, y_train_subset, atm_X_validation, y_validation, imbalance_adjusted = True)

# Train and test
train_and_test(atm_X_train, y_train, atm_X_test, y_test, atm_optimal_depth_bal, imbalance_adjusted = True)
