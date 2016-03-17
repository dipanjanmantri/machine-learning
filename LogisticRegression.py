import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
#pi=1/(1+exp(-(b0+b1*x)))

x1=np.array([1,1.1,1.2,4,5,6,7,8])
y1=np.array([1,1,1,1,1,1,1,1])
x2=np.array([7,8,9,10,11,12])
y2=np.array([2,2,2,2.0001,2.0002,2])
X=np.array([[0],[0.25],[0.5],[0.75],[1],[1.25],[1.5],[1.75]])
Y=np.array([0,0,0,0,3,3,3,3])
plt.axis([-2,12,-2,4])
classifier=LogisticRegression()
classifier.fit(X,Y)
def model(classifier,x):
    return 1/1+np.exp(-(classifier.intercept_+classifier.coef_*x))

prediction_1=classifier.predict_proba(0)
prediction_2=classifier.predict_proba(0.25)
prediction_3=classifier.predict_proba(0.5)
prediction_4=classifier.predict_proba(0.75)
prediction_5=classifier.predict_proba(1)
prediction_6=classifier.predict_proba(1.25)
prediction_7=classifier.predict_proba(1.5)
prediction_8=classifier.predict_proba(1.75)
prediction_9=classifier.predict_proba(1.555)

print(prediction_1)
print(prediction_2)
print(prediction_3)
print(prediction_4)
print(prediction_5)
print(prediction_6)
print(prediction_7)
print(prediction_8)
print(prediction_9)

print()

for i in range(-2,12):
    plt.plot(i,model(classifier,i),'ro',color='green')

#plt.plot(x1,y1,'ro',color='red')
#plt.plot(x2,y2,'ro',color='blue')
plt.plot(x1,y1,'ro',color='blue')
plt.show()
