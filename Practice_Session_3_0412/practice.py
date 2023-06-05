import numpy as np
import pandas as pd
# import seaborn as sb
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import plot_tree

# df_housing = pd.read_csv(r"California_Housing.csv")

# unique_ocean_prox = df_housing['ocean_proximity'].unique()
# print(unique_ocean_prox)

# ocean_proximity = {"NEAR BAY": 0, "<1H OCEAN": 1,
#                    "INLAND": 2, "NEAR OCEAN": 3, "ISLAND": 4}

# df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace(
#     ocean_proximity)

# print(df_housing.head())

# sb.heatmap(df_housing)
# plt.show()

df = pd.read_csv('California_Housing.csv')
