import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
trainingData=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

print("\n".join(trainingData.data[0].split("\n")[:10]))
print("Target is :",trainingData.target_names[trainingData.target[0]])

print()
countVector=CountVectorizer()
XTrainCounts=countVector.fit_transform(trainingData.data)

print(countVector.vocabulary_.get(u'software'))

t_Transformer=TfidfTransformer()
x_Transformer=t_Transformer.fit_transform(XTrainCounts)

model=MultinomialNB().fit(x_Transformer,trainingData.target)

new=['This has nothing to od with the church or the religion','Software engineering is getting hotter and hotter nowadays']
xNewCounts=countVector.transform(new)
xNewTF=t_Transformer.transform(xNewCounts)

prediction=model.predict(xNewTF)

for doc, category in zip(new, prediction):
    print((doc,trainingData.target_names[category]))