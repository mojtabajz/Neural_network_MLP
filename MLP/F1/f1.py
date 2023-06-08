import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


def func(domain):
    pi=math.pi
    result = [ i+6 for i in domain]
    # result = [ (i*2+i*3)/2 for i in domain]
    # result = [math.sin(i/7)*5 for i in domain]
    # result = [ math.sin(2*i*pi)+math.sin(5*i*pi) for i in domain]

    return result

trainpoints = 300
traindomain = 100
x_train = np.linspace(-traindomain, traindomain, trainpoints).reshape(-1,1)
y_train = np.array(func(x_train)).reshape( -1)

testpoints = 100
testdomain = 300
x_test = np.linspace(-testdomain, testdomain, testpoints).reshape(-1, 1)
y_test = np.array(func(x_test)).reshape( -1)

number_of_iteration = 1000
hidden_layer = (10)
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

fig.suptitle("trp: "    +str(trainpoints)+ 
             " , trd: " +str(traindomain)+ 
             " , tep: " +str(testpoints)+ 
             " , ted: " +str(testdomain)+ 
             "\nnitr: " +str(number_of_iteration)+ 
             " , hl: "  +str(hidden_layer)+
             " , f: i+6" )

ax.set_title('The mean squared error: ' + str(round(MSE,1)))
ax.legend(handles=[train_plt, test_plt, expected_plt])
name = "MSE " +str(round(MSE,1))+ '.png'
plt.savefig(name)
plt.show()