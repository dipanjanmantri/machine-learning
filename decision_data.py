import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data=pd.read_csv("E:\\data.txt")

#print(data.head())
#print(data.describe())
#print(data.corr())

features=data[["length","breadth","height","area"]]
target_v=data.area

feature_train, feature_test, target_train, target_test = train_test_split(features, target_v, test_size=2)

model=DecisionTreeClassifier()
predictions = model.fit(feature_train, target_train).predict(feature_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
