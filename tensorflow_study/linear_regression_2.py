import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

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

X = tf.placeholder(tf.float32, shape=(len(X_trainlist)))
Y = tf.placeholder(tf.float32, shape=(len(Y_trainlist)))

a = tf.Variable(2.0)
b = tf.Variable(1.0)

Ypred = tf.add(tf.multiply(X,a),b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pred = sess.run(Ypred,feed_dict={X:X_trainlist})
    plt.plot(X_trainlist, pred)
    plt.plot(X_trainlist, Y_trainlist, 'ro')
    plt.draw()
    plt.pause(1) # <-------
    input("Press any key to continue")
    plt.close()

    losses = []
    loss = tf.reduce_mean(tf.squared_difference(Ypred,Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss)

    for k in range(50000):
        _, _l,_a,_b= sess.run([train,loss,a,b],feed_dict={X:X_trainlist,Y:Y_trainlist})
        losses.append(_l)
        if((k+1)%500==0):
            print("Step: "+ str(k+1))
            print("a is: "+ str(_a))
            print("b is: "+ str(_b))
            print("Loss: "+ str(_l))
    plt.plot(losses)
    plt.draw()
    plt.pause(1) # <-------
    input("Press any key to continue")
    plt.close()
