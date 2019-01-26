import csv

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
    print("Total training data: "+ str(train_id-1)) ### minus the first row which is not a data

with open('./random-linear-regression/test.csv',newline = '') as train_file:
    reader = csv.reader(train_file)
    test_id = 0;
    for row in reader:  ### each row is a list with x and y
        if(test_id == 0):
            test_id+=1
            print(row[0],row[1])    ### fist row is description
        else:
            if(len(row)==2):
                test_id+=1
                print(float(row[0]), float(row[1]))
    print("Total testing data: "+ str(test_id-1))
