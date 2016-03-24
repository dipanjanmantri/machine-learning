import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x1=np.array([1,1.5,2,2.5])
y1=np.array([2,1,3,2.6])

x2=np.array([4,4.5,5,5.5])
y2=np.array([0.9,1.6,1.2,4.3])

x3=np.array([[1,0.5],[2,1.2],[3,1.8],[4,5.6]])
y3=np.array([0,0,1,1])

plt.plot(x1,y1,'ro',color='red')
plt.plot(x2,y2,'ro',color='blue')

plt.plot(4,9,'ro',color='green')

plt.axis([-4,10,-2,10])

classifier = KNeighborsClassifier(n_neighbors=2)

classifier.fit(x3,y3)

prediction=classifier.predict([4,9])

print(prediction)

plt.show()

