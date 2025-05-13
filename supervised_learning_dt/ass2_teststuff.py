import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
weather_level = {  
    "Clear" : 1,
    "Strong winds" : 2,
    "Raining" : 3,
    "Dust" : 4,
    "Smoke" : 5,
    "Fog" : 6,
    "Snowing" : 7,
}

road_level = {  
    "Dry" : 1,
    "Wet" : 2,
    "Muddy" : 3,
    "Snowy" : 4,
    "Icy" : 5,
}

def get_avg_index(desc, level):
    conditions = [c.strip() for c in desc.split(',')]
    indices = [level.get(cond, None) for cond in conditions]
    indices = [i for i in indices if i is not None] 
    return round(sum(indices) / len(indices), 2) if indices else None

def merge_atmosphere():
    atmosphere = pd.read_csv('atmospheric_cond.csv')
    atmosphere = atmosphere[atmosphere['ATMOSPH_COND_DESC'] != 'Not known']

    atmosphere_sorted = atmosphere.sort_values(by=['ACCIDENT_NO', 'ATMOSPH_COND_SEQ'])

    merged = atmosphere_sorted.groupby('ACCIDENT_NO').agg({
        'ATMOSPH_COND_DESC': lambda x: ', '.join(x)
    }).reset_index()

    merged['ATMOSPH_INDEX'] = [get_avg_index(i, weather_level) for i in merged['ATMOSPH_COND_DESC']]
    merged.to_csv("merged_atmospheric.csv", index=False)

    return

def merge_road():
    road_surface = pd.read_csv('road_surface_cond.csv')
    road_surface = road_surface[road_surface['SURFACE_COND_DESC'] != 'Unk.']

    road_surface_sorted = road_surface.sort_values(by=['ACCIDENT_NO', 'SURFACE_COND_SEQ'])

    merged = road_surface_sorted.groupby('ACCIDENT_NO').agg({
        'SURFACE_COND_DESC': lambda x: ', '.join(x)
    }).reset_index()

    merged['SURFACE_INDEX'] = [get_avg_index(i, road_level) for i in merged['SURFACE_COND_DESC']]
    merged.to_csv("merged_road_surface.csv", index=False)

    return
def merge_accident():

    merge_atmosphere()
    merge_road()
    accident = pd.read_csv('accident.csv')
    merged_atmosphere = pd.read_csv('merged_atmospheric.csv')
    merged_road_surface = pd.read_csv('merged_road_surface.csv')
    
    merged_accident = accident.merge(merged_atmosphere[['ACCIDENT_NO', 'ATMOSPH_INDEX', 'ATMOSPH_COND_DESC']], on='ACCIDENT_NO', how='left')
    merged_accident = merged_accident.merge(merged_road_surface[['ACCIDENT_NO', 'SURFACE_INDEX', 'SURFACE_COND_DESC']], on='ACCIDENT_NO', how='left')
    
    merged_accident = merged_accident[
        merged_accident['ATMOSPH_COND_DESC'].notna() & 
        merged_accident['SURFACE_COND_DESC'].notna()
    ]

    merged_accident.to_csv("merged_accident.csv", index=False)
    return

merge_accident()

# Print merged csv for testing
merged_df = pd.read_csv("merged_accident.csv")
merged_df.to_csv("merged_test.csv", index=False)


# Supervised Machine Learning 1: Decision Tree Classifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score

# Remap severity to flip order
merged_df['SEVERITY'] = merged_df['SEVERITY'].apply(lambda x: 5 - x)

def split_df(df):
    train_size = int(0.8 * len(df))
    train = df
    test = df[train_size:]
    
# Split data into 80/20 train/test
split_df(merged_df)

"""
train_size = int(0.8 * len(merged_df))
train = merged_df[:train_size]
test = merged_df[train_size:]

train_subset_size = int(0.8*len(train))
train_subset = train[:train_subset_size]
validation = train[train_subset_size:]
"""


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



