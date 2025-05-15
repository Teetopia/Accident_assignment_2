import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

accident = pd.read_csv('merged_accident.csv')

def heat_map():
    print(accident[['SEVERITY_INDEX', 'ATMOSPH_INDEX', 'SURFACE_INDEX']].corr(method = 'spearman'))
    sns.heatmap(accident[['SEVERITY_INDEX', 'ATMOSPH_INDEX', 'SURFACE_INDEX']].corr(method = 'spearman'),annot=True, annot_kws={"size": 24})
    plt.yticks(rotation=0)
    plt.show()

    print(accident[['SEVERITY_INDEX', 'ATMOSPH_INDEX', 'SURFACE_INDEX']].corr())
    sns.heatmap(accident[['SEVERITY_INDEX', 'ATMOSPH_INDEX', 'SURFACE_INDEX']].corr(),annot=True, annot_kws={"size": 24})
    plt.yticks(rotation=0)
    plt.show()

def linear_regression():
    X1 = accident[['ATMOSPH_INDEX']].values
    X2 = accident[['SURFACE_INDEX']].values
    y = accident['SEVERITY_INDEX'].values

    model1 = LinearRegression().fit(X1, y)
    y_pred1 = model1.predict(X1)

    model2 = LinearRegression().fit(X2, y)
    y_pred2 = model2.predict(X2)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X1, y, alpha=0.5, label="Data Points")
    plt.plot(X1, y_pred1, color='red', label='Regression Line')
    plt.xlabel('ATMOSPH_INDEX (Weather Severity)')
    plt.ylabel('SEVERITY_INDEX') 
    plt.title('ATMOSPH_INDEX vs SEVERITY')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X2, y, alpha=0.5, label="Data Points")
    plt.plot(X2, y_pred2, color='red', label='Regression Line')
    plt.xlabel('SURFACE_INDEX (Road Severity)')
    plt.ylabel('SEVERITY_INDEX')
    plt.title('SURFACE_INDEX vs SEVERITY')
    plt.legend()

    plt.tight_layout()
    plt.show()

heat_map()
linear_regression()