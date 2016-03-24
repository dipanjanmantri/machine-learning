import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

XBlue=np.array([1,2,3,4])
YBlue=np.array([1.5,0.5,4.4,3.4])

XRed=np.array([6,7,8,9])
YRed=np.array([2.5,1.9,5.6,4.2])

X=np.array([[1,1.5],[2,0.5],[3,4.4],[4,3.4],[6,2.5],[7,1.9],[8,5.6],[9,1.2]])
Y=np.array([0,0,0,0,1,1,1,1])

plt.plot(XBlue,YBlue,'ro',color='blue')
plt.plot(XRed,YRed,'ro',color='red')
plt.plot(5,4,'ro',color='green',markersize=10)

plt.axis([-2,10,-2,10])
classifier=GaussianNB()

classifier.fit(X,Y)

prediction=classifier.predict([5,4])

print(prediction)

plt.show()