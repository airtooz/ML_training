import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10,6)

X_trainlist = []
Y_trainlist = []
X_testlist = []
Y_testlist = []


with open('./datasets-for-isrl/Advertising.csv',newline = '') as f:
    reader = csv.reader(f)
    train_id = 0;

    for row in reader:  ### each row is a list with x and y
        if(train_id == 0):
            train_id+=1 # First row: Description about data.
        else:
            train_id+=1
            print(row)
            
            #X_trainlist.append(float(row[0]))
            #Y_trainlist.append(float(row[1]))
    print("Total training data: "+ str(train_id-1)) ### minus the first row which is not a data
