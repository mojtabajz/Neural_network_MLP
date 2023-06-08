import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def func(domain):
    result = []
    # for i in domain:
    #     result.append((i[0]+2i[1]))

    for i in domain:
        result.append([((i[0]/2)+i[1])**3 , ((i[0]/3)-i[1])**2])

    # for i in domain:
    #     result.append([(i[0]+(2*i[1])) , (i[0]-(2*i[1]))])

    return result

train_points_x = 100
train_domain_x = 50

train_points_y = 100
train_domain_y = 200
x_t = np.linspace(-train_domain_x, train_domain_x, train_points_x)
y_t = np.linspace(-train_domain_y, train_domain_y, train_points_y)
x_train = np.transpose(np.array((x_t,y_t)))
z_train = np.array(func(x_train))

test_points_x = 200
test_domain_x = 150

test_points_y = 200
test_domain_y = 250
x_s = np.linspace(-test_domain_x, test_domain_x, test_points_x)
y_s = np.linspace(-test_domain_y, test_domain_y, test_points_y)
x_test = np.transpose(np.array((x_s,y_s)))
z_test = np.array(func(x_test))

number_of_iteration = 4000
hidden_layer = (10,10,10,10,10)
trained_netwoek = MLPRegressor( hidden_layer_sizes= hidden_layer,max_iter=number_of_iteration,random_state=1,shuffle=True).fit(x_train, z_train)

z_result = trained_netwoek.predict(x_test)

summation = 0
for i in range (0,len(z_test)):
  difference = z_test[i] - z_result[i]
  squared_difference = difference**2 
  summation = summation + squared_difference
MSE = summation/len(z_test)

print('\nThe mean squared error: ' + str(round(MSE[0],1)))
