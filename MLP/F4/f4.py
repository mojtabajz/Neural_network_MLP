import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


x_train = np.array([-3,-1,1,2,4,5,6,7,9,10,11,13,14,15,17,18,20,21,22,24,25,27,28]).reshape( -1 ,1)
y_train = np.array([11 , 14 , 14.5 , 14.6  , 14.1 , 14 , 15.9 , 16 , 15.5 , 15 , 14.3 , 
            15 , 15.1 , 15.3 , 15.4 , 15.4 , 14 , 14.1 , 14.2 , 14 , 13.8 , 14.5 , 15.7 ]).reshape( -1 ,1)


x_test = np.array([-2,0,3,8,12,16,19,23,26,29]).reshape( -1 ,1)
y_test = np.array([12,14.2,14.5,15.8,15,15,15.2,14.1,14,12]).reshape( -1 ,1)


number_of_iteration = 10000
hidden_layer = (100,100,100)
trained_netwoek = MLPRegressor( hidden_layer_sizes= hidden_layer,max_iter=number_of_iteration,random_state=1,shuffle=True).fit(x_train, y_train)

y_result = trained_netwoek.predict(x_test)

summation = 0
for i in range (0,len(y_test)):
  difference = y_test[i] - y_result[i]
  squared_difference = difference**2 
  summation = summation + squared_difference
MSE = summation/len(y_test)

fig, ax = plt.subplots()
train_plt, = plt.plot(x_train, y_train, label='Train',  linewidth=3, linestyle=':')
test_plt,  = plt.plot(x_test, y_result, label='Test')
expected_plt,  = plt.plot(x_test, y_test, label='Expected_result')

fig.suptitle("trp: 24"    + 
             " , trd: 60%(-3,29)" +
             " , tep: 9" +
             " , ted: 40%(-3,29)" + 
             "\nnitr: " +str(number_of_iteration)+ 
             " , hl: "  +str(hidden_layer))

ax.set_title('The mean squared error: ' + str(round(MSE[0],1)))
ax.legend(handles=[train_plt, test_plt, expected_plt])
name = "MSE " +str(round(MSE[0],1))+ '.png'
plt.savefig(name)
plt.show()