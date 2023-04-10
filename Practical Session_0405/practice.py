import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_labor = pd.read_csv(r"LaborSupply1988.csv")

print(df_labor.shape)
print()

print(df_labor.info())
print()

print(df_labor.head())
print()

df_sample = df_labor.sample(n=10)

print('age_max: ' + str(df_labor['age'].max()))
print()

print('age_min: ' + str(df_labor['age'].min()))
print()

age_range = df_labor['age'].max() - df_labor['age'].min()
print('age_(max-min): ' + str(age_range))
print()

# Way 1/3
children = df_labor.loc[df_labor['age'] == 40, 'kids']
print(children.mean(skipna=True))  # this works
print(type(children))  # children is a Series object

# doesn't work because of 'if' part
# children2 = [x for x in df_labor['kids']
#              if age in df_labor['age'] == 40]
# print('data type of children2: ', end='')
# print(type(children2))
# print(children2)


# Way 2/3
children2 = df_labor.loc[df_labor['age'] == 40, 'kids'].tolist()
print(np.mean(children2))
print(type(children2))  # children2 is a list

# Way 3/3
age_children = list(zip(df_labor['age'], df_labor['kids']))

kids = []


def kid(a, b):
    if a == 40:
        kids.append(b)

    else:
        pass


for a, b in age_children:
    kid(a, b)

avg_kids = np.mean(kids)

print(avg_kids)
