import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer

accident = pd.read_csv('merged_accident.csv')
equal_width = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')

# a transformed version of 
accident['ATMOSPH_BIN'] = equal_width.fit_transform(accident[['ATMOSPH_INDEX']]).astype(int)
accident['SURFACE_BIN'] = equal_width.fit_transform(accident[['SURFACE_INDEX']]).astype(int)
accident['SEVERITY_BIN'] = accident['SEVERITY'] - 1

accident.hist(column=['ATMOSPH_BIN', 'SEVERITY_BIN', 'SURFACE_BIN'], bins=4)
plt.show()

def compute_probability(col):
    return col.value_counts() / col.shape[0]

def compute_entropy(col):
    probabilities = compute_probability(col)
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy

def compute_conditional_entropy(x, y):
    probability_x = compute_probability(x)
    
    temp_df = pd.DataFrame({'X': x, 'Y': y})
    
    entropy_by_group = temp_df.groupby('X')['Y'].aggregate(compute_entropy)
    conditional_entropy = sum(probability_x * entropy_by_group)
    
    return conditional_entropy

def NMI(x, y):
    # H(X) and H(Y)
    entropy_x = compute_entropy(x)
    entropy_y = compute_entropy(y)
    
    # H(Y|X)
    conditional_entropy = compute_conditional_entropy(x, y)
    
    # MI(X, Y)
    MI = entropy_y - conditional_entropy 
    # same as MI = entropy_x - compute_conditional_entropy(y, x)
    
    return MI / min(entropy_x, entropy_y)

print("NMI of weather vs severity", NMI(accident['ATMOSPH_BIN'], accident['SEVERITY_BIN']))
print("NMI of road condition vs severity", NMI(accident['SURFACE_BIN'], accident['SEVERITY_BIN']))
print("NMI of weather vs road condition", NMI(accident['ATMOSPH_BIN'], accident['SURFACE_BIN']))