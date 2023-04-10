import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# r"..." is a string object where a backslash("\") is not interpreted as an escape character
df_train = pd.read_csv(r"train.csv", index_col=0)

# .shape -> a tuple with the number of rows and the number of columns (x, y)
print(df_train.shape)
print(df_train.head())

df_test = pd.read_csv(r"test.csv", index_col=0)
print(df_test.shape)
print(df_test.head())

print(df_train.describe())


# sb.countplot(x="Survived", data=df_train)
# plt.show()

# sb.histplot(x="Age", hue="Survived", data=df_train, multiple="stack", bins=30)
# plt.show()

# sb.scatterplot(x="Age", y="Fare", hue="Survived", data=df_train)
# plt.show()

# sb.violinplot(x="Pclass", y="Fare", hue="Survived", data=df_train)
# plt.show()


# print(df_train.groupby(by="Embarked").mean())

# # sb.barplot(x="Sex", y="Age", data=df_train)
# # plt.show()


# # Survival depending on sex
# sb.histplot(x="Sex", hue="Survived", multiple="stack",
#             binwidth=1,  data=df_train)  # How can you make the bins narrower?
# plt.show()


# # Survival depending on the number of siblings
# sb.histplot(x="SibSp", hue="Survived", multiple="stack", data=df_train)
# plt.show()


# # Survival depending on the number of parents/children
# sb.histplot(x="Parch", hue="Survived", multiple="stack", data=df_train)
# plt.show()


print(df_train.info())


prediction = np.random.randint(2, size=len(df_test))
print(prediction)


# Create a DataFrame object with one column with column name "Survived"
df_test_prediction = pd.DataFrame(
    prediction, index=df_test.index, columns=["Survived"])

print(df_test_prediction.head())


df_test_prediction.to_csv(r"prediction_random.csv")

prediction = np.zeros(len(df_test), dtype=int)  # create an array with 0

df_test_prediction = pd.DataFrame(prediction, index=df_test.index, columns=[
                                  "Survived"])  # turn the array into a dataframe

# save the dataframe as an csv file
df_test_prediction.to_csv("prediction_alldead.csv")


# filter female passengers
print()
print('***Filter female passengers***')


# create a data frame with only female passengers
df_test_female = df_test.loc[df_test['Sex']
                             == 'female', :]  # ':' is a placeholder
print(df_test_female.head())


# df_test_female_index = df_test_female.index

# extract only index from the data frame => it will be an empty data frame with only the index column
df_test_female = pd.DataFrame(index=df_test_female.index)


print()
print('***Filtered female passenger index***')
print(df_test_female.head())


# print(df_test.columns)

# add a new column and fill the values depending on an existing column
df_test['Survived'] = np.where(df_test['Sex'] == 'female', 1, 0)

# df_test_prediction = df_test.loc[:, [df_test.index, 'Survived']] # doesn't work


df_test_prediction = pd.DataFrame(
    df_test['Survived'], index=df_test.index, columns=["Survived"])


print(df_test_prediction.head())

df_test_prediction.to_csv("prediction_female.csv")


print('***End***')
