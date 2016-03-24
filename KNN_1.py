import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#x-sugar
#y-salt

XFruit=np.array([9,8,7,6])
YFruit=np.array([5,4,3,2])

XVegetable=np.array([1,2,3,4])
YVegetable=np.array([4,5,6,7])

X=np.array([[9,5],[8,4],[7,3],[6,2],[1,4],[2,5],[3,6],[4,7]])
Y=np.array([0,0,0,0,1,1,1,1])

plt.plot(XFruit,YFruit,'ro',color='blue')
plt.plot(XVegetable,YVegetable,'ro',color='red')

plt.plot(7,9,'ro',color='green',markersize=12)

plt.axis([-2,10,-2,10])

classifier=KNeighborsClassifier(n_neighbors=2)

classifier.fit(X,Y)

prediction=classifier.predict([7,9])

print(prediction)

plt.show()