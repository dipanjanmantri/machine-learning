import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree

x1=np.array([1,1.5,2,2.5])
y1=np.array([2.5,5,1.2,6.9])

x2=np.array([3.5,4,4.5,5])
y2=np.array([9,12,4,6])

X=np.array([[1,2.5],[1.5,5],[2,1.2],[2.5,6.9],[3.5,9],[4,12],[4.5,4],[5,6]])
Y=np.array([0,0,0,0,1,1,1,1])

plt.plot(x1,y1,'ro',color='red')
plt.plot(x2,y2,'ro',color='blue')

plt.plot(3,8,'ro',markersize='20',color='orange')

plt.axis([0,14,0,20])

classifier=tree.DecisionTreeClassifier()

classifier.fit(X,Y)

prediction=classifier.predict([3,8])

print(prediction)

plt.show()