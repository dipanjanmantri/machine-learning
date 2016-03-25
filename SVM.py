import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

XBlue=np.array([1,1.5,2,2.5])
YBlue=np.array([0.5,3.4,1.9,2.2])

XRed=np.array([3.5,4,4.5,5])
YRed=np.array([1.2,5.6,4.2,6.5])

X=np.array([[1,0.5],[1.5,3.4],[2,1.9],[2.5,2.2],[3.5,1.2],[4,5.6],[4.5,4.2],[5,6.5]])
Y=np.array([0,0,0,0,1,1,1,1])

classifier=svm.SVC()

classifier.fit(X,Y)

print(classifier.predict([2.5,2]))

plt.axis([-2,10,-2,10])

plt.plot(XBlue,YBlue,'ro',color='blue')
plt.plot(XRed,YRed,'ro',color='red')

plt.plot(2.5,2,'ro',color='green')

plt.show()