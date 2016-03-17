from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

x=np.array([1,2,3,4,5,6])
y=np.array([1.1,1.15,1.2,1.3,1.35,.14])
y1=np.array([4.1,4.15,4.2,4.3,4.35,4.4])
plt.axis([-2,6,0,8])
plt.plot(x,y,'ro',color='black')
plt.plot(x,y1,'ro',color='blue')
plt.show()
