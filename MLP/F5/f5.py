import numpy as np
import os
from PIL import Image
from sklearn.neural_network import MLPClassifier

directory = "F5/images/train"
x_train = []
for filename in os.listdir(directory):
    train_image = Image.open(os.path.join(directory, filename))
    train_image_array = np.array(train_image)
    train_image_array = train_image_array.reshape(256)
    x_train.append(train_image_array)
x_train = np.array(x_train)
train_files = os.listdir(directory)
y_train = []
for i, file_name in enumerate(train_files):
    y_train.append(int(file_name[0]))
y_train = np.array(y_train)


directory = "F5/images/test"
x_test = []
for filename in os.listdir(directory):
    test_image = Image.open(os.path.join(directory, filename))
    test_image_array = np.array(test_image)
    test_image_array = test_image_array.reshape(256)
    x_test.append(test_image_array)
x_test = np.array(x_test)
y_train = np.array(y_train)
test_files = os.listdir(directory)
y_test = []
for i, file_name in enumerate(test_files):
    y_test.append(int(file_name[0]))
y_test = np.array(y_test)

number_of_iteration = 1000
hidden_layer = (40,40,40)
trained_netwoek = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=number_of_iteration,random_state=1,shuffle=True).fit(x_train, y_train)
y_result = trained_netwoek.predict(x_test)

count = 0 
for i in range(len(y_result)):
    if y_test[i]==y_result[i] :
        count += 1

count = (count/len(y_result))*100

print("Similarity percentage: " + str(round(count,2)) + " %")