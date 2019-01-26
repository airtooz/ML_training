import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   ### Add patch in matplotlib

###################################################################
### Plot the training data and the testing data with matplotlib ###
###################################################################

plt.rcParams['figure.figsize'] = (10,6)

X_trainlist = []
Y_trainlist = []
X_testlist = []
Y_testlist = []


with open('./random-linear-regression/train.csv',newline = '') as train_file:
    reader = csv.reader(train_file)
    train_id = 0;
    for row in reader:  ### each row is a list with x and y
        if(train_id == 0):
            train_id+=1
            print(row[0],row[1])    ### fist row is description
        else:
            if(len(row)==2):
                train_id+=1
                print(float(row[0]), float(row[1]))
                X_trainlist.append(float(row[0]))
                Y_trainlist.append(float(row[1]))
    print("Total training data: "+ str(train_id-1)) ### minus the first row which is not a data

X_train = np.asarray(X_trainlist)
Y_train = np.asarray(Y_trainlist)

plt.scatter(X_train,Y_train,s = 10, c = 'b', marker = 'o', )
plt.ylabel('Y')
plt.xlabel('X')
plt.draw()
plt.pause(1) # <-------
input("Press any key to continue")
plt.close()


with open('./random-linear-regression/test.csv',newline = '') as test_file:
    reader = csv.reader(test_file)
    test_id = 0;
    for row in reader:  ### each row is a list with x and y
        if(test_id == 0):
            test_id+=1
            print(row[0],row[1])    ### fist row is description
        else:
            if(len(row)==2):
                test_id+=1
                print(float(row[0]), float(row[1]))
                X_testlist.append(float(row[0]))
                Y_testlist.append(float(row[1]))
    print("Total testing data: "+ str(test_id-1))

X_test = np.asarray(X_testlist)
Y_test = np.asarray(Y_testlist)

plt.scatter(X_test,Y_test,s = 10, c = 'b', marker = 'o', )
plt.ylabel('Y')
plt.xlabel('X')
plt.draw()
plt.pause(1) # <-------
input("Press any key to continue")
plt.close()


##########################################
### Start process of linear regression ###
##########################################

### Suppose our best fit line is Y = aX+b

# Initial guess if our line
a = tf.Variable(2.0)
b = tf.Variable(5.0)
y = a*X_train+b

### reduce_mean: This function finds the mean of a multidimensional tensor, and the result can have a diferent dimension.
loss = tf.reduce_mean(tf.square(y-Y_train))

### input learning rate for optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

### Initializing variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

### the parameters for updated a and b in every 5000 epoch
train_param = []

for step in range(100000):
    evals = sess.run([train,a,b])[1:]
    if step % 5000 == 0:
        print(step, evals)
        print("Loss is: ",sess.run(loss))
        train_param.append(evals)


converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_param:
    cb += 1.0 / len(train_param)
    cg -= 1.0 / len(train_param)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(X_train)
    line = plt.plot(X_train, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(X_train, Y_train, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()
