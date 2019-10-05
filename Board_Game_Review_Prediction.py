import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
import seaborn as sns

dataset=pd.read_csv(r'C:\Users\USER\Desktop\Dataset\games.csv')
print(dataset.columns)
print(dataset.shape)
plt.hist(dataset["average_rating"])

print(dataset[dataset["average_rating"]==0].iloc[0])
print(dataset[dataset["average_rating"]>0].iloc[0])

dataset=dataset[dataset["users_rated"]>0]

dataset=dataset.dropna(axis=0)

plt.hist(dataset["average_rating"])

plt.show()

corrmat=dataset.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True);
plt.show()

columns=dataset.columns.tolist()
columns=[c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]
target=dataset["average_rating"]
print(target)

from sklearn.model_selection import train_test_split
train=dataset.sample(frac=0.8,random_state=1)
test=dataset.loc[~dataset.index.isin(train.index)]
print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(train[columns],train[target])

from sklearn.metrics import mean_squared_error
predictions=model.predict(test[columns])
mean_squared_error(predictions,test[target])

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimator=100,min_samples_leaf=10,random_state=1)
model.fit(train[columns],train[target])
predictions=model.predict(test[columns])
mean_squared_error(predictions,test[target])

