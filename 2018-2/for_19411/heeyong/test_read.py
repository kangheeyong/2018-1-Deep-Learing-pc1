import numpy as np



data = np.loadtxt("data2.txt")

print(data)

X_train = data[0:20,0:4]
Y_train = data[0:20,4]

X_test = data[20:27,0:4]
Y_test = data[20:27,4]

print(X_train[1:2,:].shape)
print(Y_train[2:3,].shape)

